# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import math



%matplotlib inline





data_file = "/kaggle/input/chess/games.csv"



data = pd.read_csv(data_file)

data.head()
data = data.drop(['rated', 'created_at', 'last_move_at', 'opening_ply', 'opening_name', 'white_id', 'black_id', 'increment_code', 'opening_eco'], axis=1)

data.head()
data.describe()
print("Number of games with less than 10 moves: {}".format(len(data[data['turns'] <= 10])))

data = data[data['turns'] > 10]



# Create a new column for average rating of the two users

data['rating'] = data.apply(lambda row: (row['white_rating'] + row['black_rating']) / 2, axis=1)

data = data.drop(['white_rating', 'black_rating'], axis=1)

data.head()
max_rating = data['rating'].max()

min_rating = data['rating'].min()

delta = int((max_rating - min_rating) / 3)



split_ratings = [min_rating + 1 + (i + 1) * delta for i in range(3)]



split_ratings

print(data.iloc[0]['moves'])
p = {

    "NONE": 0,

    "R1B": 1, # First black rook

    "N1B": 2, # First black knight

    "B1B": 3, # First black boshop

    "QB": 4, # black queen

    "KB": 5, # black king

    "B2B": 6, # Second black bishop

    "N2B": 7, # Second black knight

    "R2B": 8, # Second black rook

    "P1B": 9, # First black pawn

    "P2B": 10, # First black pawn

    "P3B": 11, # 3 black pawn

    "P4B": 12, # 4 black pawn

    "P5B": 13, # 5 black pawn

    "P6B": 14, # 6 black pawn

    "P7B": 15, # 7 black pawn

    "P8B": 16, # 8 black pawn

    

    "R1W": 17, # First black rook

    "N1W": 18, # First black knight

    "B1W": 19, # First black boshop

    "QW": 20, # black queen

    "KW": 21, # black king

    "B2W": 22, # Second black bishop

    "N2W": 23, # Second black knight

    "R2W": 24, # Second black rook

    "P1W": 25, # First black pawn

    "P2W": 26, # First black pawn

    "P3W": 27, # 3 black pawn

    "P4W": 28, # 4 black pawn

    "P5W": 29, # 5 black pawn

    "P6W": 30, # 6 black pawn

    "P7W": 31, # 7 black pawn

    "P8W": 32 # 8 black pawn

}



state = np.array([

    [ p['R1B'], p['N1B'], p['B1B'], p['QB'],  p['KB'],  p['B2B'], p['N2B'], p['R2B'] ],

    [ p['P1B'], p['P2B'], p['P3B'], p['P4B'], p['P5B'], p['P6B'], p['P7B'], p['P8B'] ],

    [ 0,        0,        0,        0,        0,        0,        0,        0        ],

    [ 0,        0,        0,        0,        0,        0,        0,        0        ],

    [ 0,        0,        0,        0,        0,        0,        0,        0        ],

    [ 0,        0,        0,        0,        0,        0,        0,        0        ],

    [ p['P1W'], p['P2W'], p['P3W'], p['P4W'], p['P5W'], p['P6W'], p['P7W'], p['P8W'] ],

    [ p['R1W'], p['N1W'], p['B1W'], p['QW'],  p['KW'],  p['B2W'], p['N2W'], p['R2W'] ]

])



state
def chess_pos_to_array_pos(chess_pos):

    letter = chess_pos[0]

    number = int(chess_pos[1])

    row = 8 - number

    col = (ord(letter) - 65)

    

    assert row >= 0 and row <= 7

    assert col >= 0 and col <= 7

    

    return (row, col)



# Testing:

print(chess_pos_to_array_pos("A1"))

print(chess_pos_to_array_pos("A8"))

print(chess_pos_to_array_pos("H1"))

print(chess_pos_to_array_pos("H8"))

def move(src, dst):

    src_pos = chess_pos_to_array_pos(src)

    dst_pos = chess_pos_to_array_pos(dst)