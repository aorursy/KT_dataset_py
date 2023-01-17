import numpy as np  # linear algebra
# Get sub matrix which remove exceptRow & exceptCol 

def get_sub_matrix(mat, exceptRow, exceptCol):

    # use list comprehension to generate list-2D

    return np.array([

        [

            mat[row][col]

            for col in range(mat.shape[1]) if col != exceptCol

        ]

        for row in range(mat.shape[0]) if row != exceptRow

    ])





def calc_det(mat):

    n = mat.shape[0]

    if n != mat.shape[1]:

        raise Exception('Matrix is not square matrix!')



    # base-case

    if n == 2:

        return mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0]



    det_of_sub_matrix_list = [

        ((-1) ** col) * mat[0][col] * calc_det(get_sub_matrix(mat, 0, col))

        for col in range(n)

    ]

    return sum(det_of_sub_matrix_list)



# transpose row to column

def get_transpose_matrix(mat):

    return np.array(

        [

            [mat[col, row] for col in range(mat.shape[0])]

            for row in range(mat.shape[1])

        ]

    )





def get_adj_matrix(mat):

    matT = get_transpose_matrix(mat)



    return np.array(

        [

            [

                ((-1) ** (row + col)) * calc_det(get_sub_matrix(matT, row, col))

                for col in range(mat.shape[1])

            ]

            for row in range(mat.shape[0])

        ]

    )





def inverse_matrix(mat):

    det_value = calc_det(mat)

    if det_value == 0:

        raise Exception("Matrix doesn't have inverse!")



    adj_matrix = get_adj_matrix(mat)

    return adj_matrix / det_value

##  or

#     for row in range(mat.shape[0]):

#         for col in range(mat.shape[1]):

#             adj_matrix[row][col] /= det_value

#     return adj_matrix



matrix = np.random.rand(3, 3)



print('matrix = ', matrix)

print('=====================\n')



print('det = ', calc_det(matrix))

print('=====================\n')



print('inverse =', inverse_matrix(matrix))

print('=====================\n')
