import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from random import randrange



def generateMatrix(n):

    matrix = []

    for i in range(n):

        row = []

        for j in range(n):

            row.append(randrange(100))

        matrix.append(row)

    matrix = np.array(matrix)

    return matrix

    

def detMatrix(matrix, k = 0, index = 0):

    shapeMatrix = matrix.shape

    result = 0

    if shapeMatrix[0] == 2:

        if index%2 == 0:

            h = 1

        else:

            h = -1

        return h*k*(matrix[0,0] * matrix[1,1] - matrix[0,1] * matrix[1,0])

    else:

        for i in range(shapeMatrix[0]):

            subMat = subMatrix(0, i, matrix)

            result += detMatrix(subMat, matrix[0, i], i)

    return result



def subMatrix(row, col, matrix):

    matrix = np.delete(matrix, row, 0) # delete row

    matrix = np.delete(matrix, col, 1) # delete col

    return matrix



def inverseMatrix(matrix):

    shapeMatrix = matrix.shape

    detMat = detMatrix(matrix)

    if detMat == 0:

        return print('This matrix can not reverse')

    # Chuyen vi ma tran

    tranposeMatrix = []

    for i in range(shapeMatrix[0]):

        tranposeMatrix.append(matrix[:,i])

    tranposeMatrix = np.array(tranposeMatrix)

    adjMatrix = calAdjMatrix(tranposeMatrix)

    k = 1/detMat

    result = []

    for i in range(shapeMatrix[0]): # row

        row = []

        for j in range(shapeMatrix[1]): # col

            row.append(adjMatrix[j,i] * k)

        result.append(row)

    result = np.array(result)

    return result

    

def calAdjMatrix(matrix):

    result = []

    shapeMatrix = matrix.shape

    for i in range(shapeMatrix[1]): # col

        row = []

        for j in range(shapeMatrix[0]): # row

            subMat = subMatrix(j, i, matrix)

            row.append(detMatrix(subMat, 1, i+j))

        result.append(row)

    result = np.array(result)

    return result

            

            
print('Matrix 3x3:\n', generateMatrix(3))
matrix = generateMatrix(3)

print('Det matrix:\n', matrix)

print('Result:', detMatrix(matrix))
matrix = generateMatrix(3)

print('Reverse matrix:\n', matrix)

print('Result:\n', inverseMatrix(matrix))