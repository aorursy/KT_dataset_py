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
# This module implements pseudo-random number generators for various distributions.

import random

# This module provides a high-performance multidimensional array object, and tools for working with these arrays.

import numpy as np







# Generate a n x n random matrix

#

# @param sequence {list, tuple, string, set} Sequence of numbers for random

# @param n {int} Dimensional number

#

# @return {numpy.ndarray} Multidimensional array

def random_matrix(sequence, n):

    # declare variables

    matrix = []

    row_index = 0



    while row_index < n:

        # return a list (length = n), numbers chosen from the sequence

        row = random.sample(sequence, n)

        

        matrix.append(row)

        # increase row index

        row_index = row_index + 1



    return np.array(matrix)







# Test

# range(10) returns [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

print(random_matrix(range(10), 3))
#LAPLACE EXPANSION



import numpy as np







# Calculate 2x2 matrix determinant

#

# @param _2x2_matrix {list, numpy.ndarray, tuple}

#

# @return {int, float}

def cal2x2MatrixDet(_2x2_matrix):

    return _2x2_matrix[0][0] * _2x2_matrix[1][1] - _2x2_matrix[0][1] * _2x2_matrix[1][0]







# Get submatrix

#

# @param row_index {int} Remove row in matrix at row_index

# @param col_index {int} Remove column in matrix at column_index

# @param matrix {array}

#

# @return {array}

def getSubmatrix(row_index, col_index, matrix):

    matrix_len = len(matrix[0])

    submatrix = []

                   

    for j in range(matrix_len):

        # Remove row

        if j == row_index:

            continue

            

        row = []



        for k in range(matrix_len):

            # Remove column

            if k == col_index:

                continue



            row.append(matrix[j][k])



        submatrix.append(row)

        

    return submatrix







# Calculate n x n matrix determinant

#

# @param matrix {list, numpy.ndarray, tuple}

#

# @return {int, float}

def calMatrixDet(matrix):

    matrix_len = len(matrix[0])

    

    if matrix_len == 2:

        return cal2x2MatrixDet(matrix)

    

    det = 0

    

    for i in range(matrix_len):

        submatrix = getSubmatrix(0, i, matrix)

        temp_det = matrix[0][i] * calMatrixDet(submatrix)

                       

        if i % 2 == 0:

            det += temp_det

        else:

            det -= temp_det

                           

    return det







# Test

print('<!-- 2x2 Matrix Determinant -->')

_list = [[4, 6], [3, 8]]

_np_array = np.array(_list)

_tuple = ((4, 6), (3, 8))

print('cal2x2MatrixDet _list :', cal2x2MatrixDet(_list))

print('cal2x2MatrixDet _np_array :', cal2x2MatrixDet(_np_array))

print('cal2x2MatrixDet _tuple :', cal2x2MatrixDet(_tuple))

print('calMatrixDet _list :', calMatrixDet(_list))

print('calMatrixDet _np_array :', calMatrixDet(_np_array))

print('calMatrixDet _tuple :', calMatrixDet(_tuple))







print('\n\n\n<!-- 3x3 Matrix Determinant -->')

_3x3_list = [[6, 1, 1], [4, -2, 5], [2, 8 ,7]]

_3x3_np_array = np.array(_3x3_list)

_3x3_tuple = ((6, 1, 1), (4, -2, 5), (2, 8 ,7))

print('calMatrixDet _3x3_list :', calMatrixDet(_3x3_list))

print('calMatrixDet _3x3_np_array :', calMatrixDet(_3x3_np_array))

print('calMatrixDet _3x3_tuple :', calMatrixDet(_3x3_tuple))







print('\n\n\n<!-- n x n Matrix Determinant -->')

_nxn_np_array = random_matrix(range(-9, 10), 5)

print(_nxn_np_array)

print('numpy.linalg.det :', np.linalg.det(_nxn_np_array))

print('calMatrixDet :', calMatrixDet(_nxn_np_array))
def detMatrix(matrix):

    matrix_len = len(matrix[0])

    det_matrix = []

    

    for i in range(matrix_len):

        row = []

        

        for j in range(matrix_len):

            submatrix = getSubmatrix(i, j, matrix)

            row_element = calMatrixDet(submatrix)

            row.append(row_element)

            

        det_matrix.append(row)



    return det_matrix

        



    

def inverse(matrix, det):

    matrix_len = len(matrix[0])

    

    for i in range(matrix_len):

        for j in range(matrix_len):

            if (i % 2) != (j % 2):

                matrix[i][j] *= (-1) / det

            else:

                matrix[i][j] *= 1 / det

                

    return matrix







def transpose(matrix):

    matrix_len = len(matrix[0])

    transpose_matrix = []

    

    for i in range(matrix_len):

        row = []

        

        for j in range(matrix_len):

            row.append(matrix[j][i])

            

        transpose_matrix.append(row)

            

    return transpose_matrix







_3x3_matrix = [[1, 2, 3], [0, 1, 4], [5, 6 ,0]]

print('_3x3_matrix :\n', np.array(_3x3_matrix))



_3x3_matrix_det = calMatrixDet(_3x3_matrix)

print('\n\n_3x3_matrix_det : ', np.array(_3x3_matrix_det))



_3x3_transpose_matrix = transpose(_3x3_matrix)

print('\n\n_3x3_transpose_matrix :\n', np.array(_3x3_transpose_matrix))



_3x3_inverse_matrix = inverse(detMatrix(_3x3_transpose_matrix), _3x3_matrix_det)

print('\n\n_3x3_inverse_matrix :\n', np.array(_3x3_inverse_matrix))