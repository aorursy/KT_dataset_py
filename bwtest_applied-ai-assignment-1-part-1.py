##Load the libraries Needed

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import cv2

import os
plt.imshow(cv2.imread("../input/aiappliedtest/assignment_1/assignment_0/connect4.png")[:,:,::-1])

plt.show()
def backread(im):

    return cv2.imread(im)[:,:,::-1]



edf = pd.read_pickle("../input/aiappliedtest/assignment_1/assignment_0/edf.pkl")

pic = backread("../input/aiappliedtest/assignment_1/assignment_0/connect4.png")

red = backread("../input/aiappliedtest/assignment_1/assignment_0/red.png")

yellow = backread("../input/aiappliedtest/assignment_1/assignment_0/yellow.png")

white = backread("../input/aiappliedtest/assignment_1/assignment_0/white.png")

Data = pd.read_csv("../input/aiappliedtest/assignment_1/assignment_0/Examples.csv")



def synthetic(mat):

    mpic = pic.copy()

    for i in edf.index:

        w = edf.loc[i]

        x1,y1 = w['left']

        f = w['f']

        if mat[w['x'],w['y']] == 0:    

            mpic[y1:y1+f,x1:x1+f] = white

        elif mat[w['x'],w['y']] == 1:

            mpic[y1:y1+f,x1:x1+f] = red

        elif mat[w['x'],w['y']]== 2:

            mpic[y1:y1+f,x1:x1+f] = yellow

        else:

            mpic[y1:y1+f,x1:x1+f] = 0

    return mpic



def print_board(board):

    f,ax = plt.subplots(figsize = (6,6))

    plt.imshow(synthetic(board))

    plt.show()

    

def loadfromarray(ar):

    ar = ar.values

    ar = ar[0][1:]

    try:

        ar = ar.reshape(6,7).astype('float')

        return ar

    except:

        ar = ar.reshape(6,7)

        return ar
example0 = np.array([  [0., 0., 0., 0., 2., 0., 0.],

                       [0., 0., 0., 0., 2., 0., 0.],

                       [0., 0., 0., 2., 2., 0., 0.],

                       [0., 0., 0., 1., 2., 0., 0.],

                       [0., 0., 0., 2., 1., 1., 0.],

                       [0., 1., 0., 1., 2., 1., 1.]  ])
example0
print_board(example0)
RedWins = loadfromarray(Data[Data.file_names == 'redwins'])

print_board(RedWins)
RedWins
YellowWins = loadfromarray(Data[Data.file_names == 'yellowwins'])

print_board(YellowWins)
Red_is_next = loadfromarray(Data[Data.file_names == 'redisnext'])

print_board(Red_is_next)
Yellow_is_next = loadfromarray(Data[Data.file_names == 'yellownext'])

print_board(Yellow_is_next)
floating_piece = loadfromarray(Data[Data.file_names == 'floating_piece'])

print_board(floating_piece)
invalid_proportion = loadfromarray(Data[Data.file_names == 'invalid_proportion'])

print_board(invalid_proportion)
corrupt_ = loadfromarray(Data[Data.file_names == 'corrupt_'])

print_board(corrupt_)
corrupt_
corrupt = loadfromarray(Data[Data.file_names == 'corrupt'])

print_board(corrupt)
corrupt