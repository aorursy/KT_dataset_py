# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import networkx as nx

import math as ma



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
seed = 3

abet = alphabet_len = seed ** 2



cells_list = [c for c in range(abet**2)]

cells = pd.Series(cells_list, dtype="category")



column_list = [c % abet for c in cells_list]

columns = pd.Series(column_list, dtype="category")



row_list = [ma.floor(c / abet) for c in cells_list]

rows = pd.Series(row_list, dtype="category")



block_list = [((ma.floor(c / seed) % abet) % seed) + (ma.floor(c / seed**3) * seed) for c in cells_list]

blocks = pd.Series(block_list, dtype="category")
frame = {'Cell': cells,'Column': columns, 'Row': rows, 'Block': blocks}

board = pd.DataFrame(frame)
rows = board.groupby('Row')['Column'].apply(list)

rows
solution_list = [((c % abet) - (ma.floor(c / seed**2) * seed) - (ma.floor(c / seed**3) * seed**2) - ma.floor(c / seed**3)) % abet + 1 for c in cells_list]
solution = pd.Series(solution_list, dtype="category")

solved_game = board.assign(Solution = solution )



solved_game.groupby(['Solution']).nunique()
solved_game.groupby('Row')['Solution'].apply(list)