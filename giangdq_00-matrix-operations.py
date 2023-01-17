import numpy as np # linear algebra

import random
# https://en.wikipedia.org/wiki/Determinant

def det(mat):

    if len(mat) == 2:

        return mat[0][0] * mat[1][1] - mat[1][0] * mat[0][1]

    else:

        det_val = 0;

        for c in range(len(mat)):

            minor = get_minor_matrix(mat, 0, c)

            det_val += mat[0][c] * det(minor) * (1 if c % 2 == 0 else -1)

        return det_val;
def transpose_matrix(mat):

    return np.matrix([[mat[j][i] for j in range(len(mat))] for i in range(len(mat))])
def get_minor_matrix(mat,i,j):

    # 1. Remove the row i-th from original matrix.

    # 2. Then, remove the colum j-th.

    # Because the input matrix is numpy matrix,

    # use concatenate() to patch the split rows/colums together.

    return [np.concatenate((row[:j], row[j+1:])) for row in (np.concatenate((mat[:i], mat[i+1:])))]
def adjugate_matrix(mat):

    size = len(mat)

    mat_adj = []

    for r in range(size):

        row = []

        for c in range(size):

            minor = get_minor_matrix(mat,r,c)

            det_val = det(minor)

            row.append(det_val if (r + c) % 2 == 0 else -1 * det_val)

        mat_adj.append(row)

    return mat_adj
# https://www.wikihow.com/Find-the-Inverse-of-a-3x3-Matrix

def inverse(mat, det_value):

    if det_value == 0:

        return 'There is no inverse matrix.'

    

    # 1. Get adjugate matrix

    # 2. Get tranpose of adjugate matrix

    # 3. Devide transpose matrix by determinant

    mat_adj = adjugate_matrix(mat)

    mat_transpose = transpose_matrix(mat_adj)

    return mat_transpose / det_value
def gen_matrix(m, n, max):

    # Randomize int matrix

    # return np.random.randint(0, max, size=(m, n))  

    

    # Randomize float matrix

    seed = random.randrange(max)

    return np.random.rand(m, n) * seed
def main():

    print('Input matrix')

    a = gen_matrix(3, 3, 1000000)

    print(a)

    print('============================')

    

    print('Determinant')

    det_value = det(a)

    print(det_value)

    print('Use numpy')

    print(np.linalg.det(a))

    print('============================')

    

    print('Inverse matrix')

    inverse_mat = inverse(a, det_value)

    print(inverse_mat)

    print('Check')

    print(a.dot(inverse_mat))

    print('============================')



main()