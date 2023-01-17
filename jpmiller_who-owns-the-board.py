import numpy as np

np.set_printoptions(edgeitems=200)

import pandas as pd

pd.set_option("display.max_columns", 200)

import string
# Make column names so that board rows are numbered top-down

numbers = np.arange(6,0,-1)

letters = list(string.ascii_lowercase[0:7])

colnames = [l+str(n) for l in letters for n in numbers]



df = pd.read_csv("../input/connect-four-datasets/connect4_openings.csv", 

                 header=None, names=colnames+['winner']).sort_index(axis=1)



outcomes = df.pop('winner')

df.head()
# Change values to integer

player_dict = {'x': 1,

               'o': -1,

               'b': 0

               }



df = df.replace(player_dict)
# Reshape

new_board = np.reshape(df.to_numpy(), (-1, 7,6))

new_board = np.swapaxes(new_board, 1, 2)

new_board[0:2]