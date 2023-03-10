!pip install python-chess
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import chess



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/train.csv')
first = df.head(1)
first['fen']
first['best']
board = chess.Board(first['fen'][0])

board
move = chess.Move.from_uci(first['best'][0])

board.san(move)
board.push(move)

board