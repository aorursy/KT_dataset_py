from pprint import pprint

import numpy as np
def gen_rand_matrix(n):

    if n < 1:

        return [1]

    

    return np.random.randint(n, size = (n, n))        
N = 3

m = gen_rand_matrix(N)

pprint(m)

print(m.shape)
def matrix_remove_index(m, i, j):

    if m.shape[0] != m.shape[1]:

        print('This is not our square matrix')

        return m

    

    if (i >= N or j >= N):

        print('Unsuficient index')

        return m

    

    return np.delete(np.delete(m, i, 0), j, 1)
pprint(matrix_remove_index(m, 0, 0))

pprint(matrix_remove_index(m, 0, 1))

pprint(matrix_remove_index(m, 2, 2))
def det(m):

    if m.shape[0] != m.shape[1]:

        print(m.shape[0], m.shape[1])

        print('This is not our square matrix')

        return 0



    #print('start calculating det for')

    #pprint(m)

    if (2, 2) == m.shape:

        determin = m[0, 0] * m[1, 1] - m[1, 0] * m[0, 1]

        #print('determin 2d:', determin)

        return determin



    length = m.shape[0]

    determin = 0

    sign = 1

    for i in range(length):

        m1 = matrix_remove_index(m, 0, i)

        #pprint(m1)

        #print('---')

        #pprint(m)

        determin = determin + sign * m[0, i] * det(m1)

        sign = - sign

        

    print('determin result:', determin)

    return determin
a = gen_rand_matrix(3)

print('=========')

print(det(a))

print('=========')

print(np.linalg.det(a))
def transpose_matrix(m):

    if m.shape[0] != m.shape[1]:

        print('This is not our square matrix')

        return m



    tranposed_matrix = np.copy(m)



    length = m.shape[0]

    for i in range(length):

        for j in range(i + 1, length):

            tmp = tranposed_matrix[i, j]

            tranposed_matrix[i, j] = tranposed_matrix[j, i]

            tranposed_matrix[j, i] = tmp

            

    return tranposed_matrix
pprint(transpose_matrix(a))

print('=========')

pprint(a.T)
def find_determine_sub_matrix(m):

    if m.shape[0] != m.shape[1]:

        print('This is not our square matrix')

        return m



    determined_matrix = np.copy(m)

    

    length = m.shape[0]

    for i in range(length):

        for j in range(length):

            sub_matrix = matrix_remove_index(m, i, j)

            determined_matrix[i, j] = det(sub_matrix)

    

    return determined_matrix
pprint(a)

print('=========')

pprint(find_determine_sub_matrix(a))
def create_cofactor_matrix(m):

    if m.shape[0] != m.shape[1]:

        print('This is not our square matrix')

        return m

    

    cofactor_matrix = np.copy(m)

    

    length = m.shape[0]

    for i in range(length):

        sign = 1

        for j in range(length):

            cofactor_matrix[i, j] = sign * cofactor_matrix[i, j]

            sign = - sign

            

    return cofactor_matrix
pprint(a)

print('=========')

pprint(create_cofactor_matrix(a))
def find_inverse_matrix(m):

    if m.shape[0] != m.shape[1]:

        print('This is not our square matrix')

        return m

    

    determine_value = det(m)

    

    if (determine_value == 0):

        print('This matrix does not have inverse matrix')

        return m

    

    cofactor_matrix = create_cofactor_matrix(m)

    

    return (1 / determine_value) * cofactor_matrix
pprint(a)

print('=========')

pprint(find_inverse_matrix(a))

print('=========')

pprint(np.linalg.inv(a))