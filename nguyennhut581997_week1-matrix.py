# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os
def sub_matrix(matrix, row, col):

    ma_row = np.delete(matrix, row, axis=0) #remove row

    ma_col = np.delete(ma_row, col, axis=1) # remove col

    return ma_col



def calc_det(matrix):

    n = matrix.shape[0]

    

    if n != matrix.shape[1]:

        raise Exception('Matrix is not square matrix!')

        

    if n == 2:

        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]



    det = 0

    for col in range(n):

        det += ((-1) ** col) * matrix[0][col] * calc_det(sub_matrix(matrix, 0, col))

        

    return det
def transpose_matrix(matrix):

    return np.array(

        [

            [matrix[col, row] for col in range(matrix.shape[0])]

            for row in range(matrix.shape[1])

        ]

    )





def adj_matrix(matrix):

    matrixT = transpose_matrix(matrix)



    return np.array(

        [

            [

                ((-1) ** (row + col)) * calc_det(sub_matrix(matrixT, row, col))

                for col in range(matrix.shape[1])

            ]

            for row in range(matrix.shape[0])

        ]

    )





def inverse_matrix(mat):

    det = calc_det(mat)

    

    if det == 0:

        raise Exception("Matrix doesn't have inverse!!!")



    adj_mat = adj_matrix(mat)

    

    return adj_mat / det
matrix = np.random.rand(3, 3)



print('\nmatrix = \n', matrix)



print('\ndet = \n', calc_det(matrix))



inverse = inverse_matrix(matrix)

print('\ninverse =\n', inverse)

print('\ncheck inverse =\n',inverse.dot(matrix))