import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import json

import os

import matplotlib.pyplot as plt

from matplotlib import colors
blk, blu, red, grn, ylw, gry, pur, orn, azu, brw = range(10)



COLOURS = [

    "#000",

    "#0074D9", # blue

    "#FF4136", # red

    "#2ECC40", # green

    "#FFDC00", # yellow

    "#AAAAAA", # grey

    "#F012BE", # fuschia

    "#FF851B", # orange

    "#7FDBFF", # teal

    "#870C25"  # brown

]

cmap = colors.ListedColormap(COLOURS)

norm = colors.Normalize(vmin=0, vmax=9)



task = "/kaggle/input/abstraction-and-reasoning-challenge/training/0a938d79.json"

with open(task, 'r') as f:

    data = json.load(f)

    

first_task = data["train"][0]



first_task_in = np.array(first_task["input"], dtype=np.uint8)

plt.imshow(first_task_in, cmap=cmap, norm=norm)

plt.show()

first_task_out = np.array(first_task["output"], dtype=np.uint8)

plt.imshow(first_task_out, cmap=cmap, norm=norm)

plt.show()
class DFSA():



    def __init__(self, state_in, state_out, transitions):

        # Initiliase start, end states and pad

        self.start_state = np.pad(state_in, (1,1), 'constant', constant_values=(0,))

        self.end_state = np.pad(state_out, (1,1), 'constant', constant_values=(0,))

        self.transitions = transitions



    def run(self):

        

        current_state = self.start_state.copy()

        

        # Symmetry: handles situation where we need to propagate the colours horizontally

        symmetry_red = np.zeros((1, 3), dtype=np.uint8)

        symmetry_azu = np.zeros((1, 3), dtype=np.uint8)

        symmetry_red[0, 0] = red

        symmetry_azu[0, 0] = azu

        next_symmetry_red = symmetry_red.copy()

        next_symmetry_azu = symmetry_azu.copy()



        next_symmetry_red[0, 2] = azu

        next_symmetry_azu[0, 2] = red

        

        

        while not np.array_equal(current_state, self.end_state):

            # Have we filled a the height of the matrix with this colour?

            if current_state[1, 8] == azu and not ''.join([str(i) for i in symmetry_red.flatten()]) in self.transitions.keys(): # if azu add symmetry

                # if yes then add symmetry transitions

                self.transitions[''.join([str(i) for i in symmetry_red.flatten()])] = next_symmetry_red

                self.transitions[''.join([str(i) for i in symmetry_azu.flatten()])] = next_symmetry_azu

            

            # Go over matrix and keep updating neighbourhoods

            for i in range(1, current_state.shape[0]-1):

                for j in range(1, current_state.shape[1]-2):

                    current_nbh = current_state[i-1:i+2, j-1:j+2]

                    if len(self.transitions.keys()) == 4:

                        current_nbh = current_nbh[1]

                        

                    if ''.join([str(i) for i in current_nbh.flatten()]) not in self.transitions.keys():

                        continue

                        

                    new_nbh = self.transitions[''.join([str(i) for i in current_nbh.flatten()])]

                    if len(self.transitions.keys()) < 4:

                        current_state[i-1:i+2, j-1:j+2] = new_nbh

                    else:

                        current_state[i, j-1:j+2] = new_nbh

        self.evolved_state = current_state

        

    def get_state(self):

        return self.evolved_state
transitions = {}

# Vertical

symbol_red = np.zeros((3, 3), dtype=np.uint8)

symbol_azu = np.zeros((3, 3), dtype=np.uint8)

symbol_red[0, 1] = red

symbol_azu[2, 1] = azu



next_red = symbol_red.copy()

next_azu = symbol_azu.copy()

next_red[1, 1] = red

next_azu[1, 1] = azu



transitions[''.join([str(i) for i in symbol_red.flatten()])] = next_red

transitions[''.join([str(i) for i in symbol_azu.flatten()])] = next_azu



dfsa = DFSA(first_task_in, first_task_out, transitions)



dfsa.run()



plt.imshow(dfsa.get_state(), cmap=cmap, norm=norm)

plt.show()