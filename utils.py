import os
import sys
import time
from copy import deepcopy

import cv2
import matplotlib.pyplot as plt
import numpy as np


class Tile():
    def __init__(self, img, idx):
        self.img = img
        self.idx = idx
        self.neighbors = {
            'top': [],
            'bottom': [],
            'left': [],
            'right': []
        }
        self.exclusions = {
            'top': [],
            'bottom': [],
            'left': [],
            'right': []
        }
        
    def __str__(self):
        plt.figure(figsize = (2, 2))
        plt.imshow(self.img)
        plt.show()
        return "Tile object"
    
    def __getitem__(self, idx):
        return self.img[idx]
        
    def add_neighbor(self, direction, tile):
        self.neighbors[direction].append(tile.idx)


def check_side(side1, side2):
    ratio = 0.8
    num_pixels = np.prod(side1.shape)
    threshold = ratio * num_pixels
    if np.sum(side1 == side2) >= threshold:
        return True
    elif np.sum(side1[:-1] == side2[1:]) >= threshold:
        return True
    elif np.sum(side1[1:] == side2[:-1]) >= threshold:
        return True


def check_and_add(tile1, tile2):
    if check_side(tile1[0, :], tile2[-1, :]):
        tile1.add_neighbor('top', tile2)
        tile2.add_neighbor('bottom', tile1)
    if check_side(tile1[-1, :], tile2[0, :]):
        tile1.add_neighbor('bottom', tile2)
        tile2.add_neighbor('top', tile1)
    if check_side(tile1[:, 0], tile2[:, -1]):
        tile1.add_neighbor('left', tile2)
        tile2.add_neighbor('right', tile1)
    if check_side(tile1[:, -1], tile2[:, 0]):
        tile1.add_neighbor('right', tile2)
        tile2.add_neighbor('left', tile1)


def reduce_prob(choices, tiles, row, col, rows, cols, TILE_IDX_LIST):
    neighbor_choices = []
    valid_choices = deepcopy(TILE_IDX_LIST)
    for i, j, direction in [[row-1, col, 'bottom'], [row+1, col, 'top'], [row, col-1, 'right'], [row, col+1, 'left']]:
        exclusion_idx_list = []
        if 0 <= i < rows and 0 <= j < cols:
            for tile_idx in choices[(i, j)]:
                tile = tiles[tile_idx]
                exclusion_idx_list.append(tile.exclusions[direction])
        total_num = len(exclusion_idx_list)
        if len(exclusion_idx_list) > 0:
            for idx in TILE_IDX_LIST:
                vote = 0
                for exclusion in exclusion_idx_list:
                    if idx in exclusion:
                        vote += 1
                if (vote == total_num) and (idx in valid_choices):
                    valid_choices.remove(idx)
    if len(valid_choices) == 0:
        return None
    else:
        choices[(row, col)] = valid_choices
        return choices


def get_min_entropy_coord(entropy_board, observed):
    rows, cols = entropy_board.shape
    min_row, min_col = -1, -1
    min_entropy = 1000
    coord_list = []
    for row in range(rows):
        for col in range(cols):
            if not observed[row, col]:
                if 1 <= entropy_board[row, col] < min_entropy:
                    min_entropy = entropy_board[row, col]
                    coord_list = []
                    coord_list.append((row, col))
                elif 1 <= entropy_board[row, col] == min_entropy:
                    coord_list.append((row, col))
    if len(coord_list) > 0:
        coord_idx = np.random.choice(np.arange(len(coord_list)))
        min_row, min_col = coord_list[coord_idx]
        return min_row, min_col
    else:
        return -1, -1


def update_entropy(choices, rows, cols):
    entropy_board = np.zeros(shape = (rows, cols))
    for row in range(rows):
        for col in range(cols):
            entropy_board[row, col] = len(choices[(row, col)])
    return entropy_board