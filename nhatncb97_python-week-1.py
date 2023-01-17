import numpy as np

import numpy.linalg as linalg



test_matrix = np.random.rand(3,3);

print('Random matrix\n', test_matrix);



def det_2x2(matrix):

    return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];

def det_3x3(m):

    return (m[0][0] * (m[1][1] * m[2][2] - m[2][1] * m[1][2])

           -m[1][0] * (m[0][1] * m[2][2] - m[2][1] * m[0][2])

           +m[2][0] * (m[0][1] * m[1][2] - m[1][1] * m[0][2]));



#Transpose the matrix

def transpose_matrix(matrix):

    tempMatrix = np.zeros((3,3));

    for i in range(len(matrix)): 

        for j in range(len(tempMatrix)): 

            tempMatrix[i][j] = matrix[j][i];

    return tempMatrix;



#Get array of all minor matrix in matrix 3x3

def minorMatrix(matrix):

    minor_matrix = [];

    for k in range(3):

        for l in range(3):

            tmp_mat = [];

            for i in range(3):

                tmp_row = []

                for j in range(3):

                    if i != k and j !=l:

                        tmp_row.append(matrix[i][j]);

                        if len(tmp_row) == 2:

                            tmp_mat.append(tmp_row);

                            tmp_row = [];

                        if len(tmp_mat) == 2:

                            minor_matrix.append(tmp_mat);

                            tmp_mat = [];

    return minor_matrix;



#Matrix created from det of minor matrix

def matrix_from_det_of_minor_matrix(minor_matrices_arr):

    adj_matrix = [];

    row = [];

    for k in range(len(minor_matrices_arr)):

        row.append(det_2x2(minor_matrices_arr[k]));

        if len(row) == 3:

            adj_matrix.append(row);

            row = [];

    return adj_matrix;



#Adj matrix

def cofactors(matrix):

    for i in range(3):

        for j in range(3):

            if (i + j)%2 != 0:

                matrix[i][j] = -matrix[i][j];

    return matrix;



#Inverse matrix

def inverse_matrix(matrix):

    det = det_3x3(matrix);

    tp_matrix= transpose_matrix(matrix);

    minor_matrices_arr = minorMatrix(tp_matrix);

    matrix_from_minor_matrix = matrix_from_det_of_minor_matrix(minor_matrices_arr);

    cofactors_matrix = cofactors(matrix_from_minor_matrix);

    inverse_det = 1/det;

    for i in range(3):

        for j in range(3):

            cofactors_matrix[i][j] = inverse_det * cofactors_matrix[i][j];

    return np.matrix(cofactors_matrix);

print('-----------------------');

print('Numpy det',linalg.det(test_matrix));

print('Det no np', det_3x3(test_matrix));

print('-----------------------');

print('Numpy inverse matrix \n',linalg.inv(test_matrix));

print('Inverse matrix no np\n', inverse_matrix(test_matrix));