# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
theBoard = {'top-L': ' ', 'top-M': ' ', 'top-R': ' ',

            'mid-L': ' ', 'mid-M': ' ', 'mid-R' : ' ',

            'low-L': ' ', 'low-M': ' ', 'low-R': ' ' }

def printBoard(board):

    print(board['top-L'] + '|' + board['top-M'] + '|' + board['top-R'])

    print('-+-+-')

    print(board['mid-L'] + '|'+  board['mid-M'] + '|' + board['mid-R'])

    print('-+-+-')

    print(board['low-L'] + '|'+  board['low-M'] + '|' + board['low-R'])

turn = "X"

for i in range(9):

    printBoard(theBoard)

    print('Turn for' + turn + '. Move on which space?')

    move = input()

    theBoard[move] = turn

    if turn == 'X':

        turn = '0'

    else:

        turn = "X"

printBoard(theBoard)