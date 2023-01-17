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
import numpy as np

import random



# Create a matrix 3x3 with random number

def createSquareMatrix(numberOfRows):

    matrix = []

    for i in range(numberOfRows):

        row = []

        for j in range(numberOfRows):

            row.append(random.randint(-100,100))



        matrix.append(row)

        

    return np.array(matrix)

    

print(createSquareMatrix(3))
import numpy as np

import random



# Calculate determinant (det) for a matrix

# References: https://www.ieev.org/2011/09/ma-tran-matrix-va-inh-thuc-determinant.html



# 1. Create a matrix

### copy above function to here...

def createSquareMatrix(numberOfRows):

    matrix = []

    for i in range(numberOfRows):

        row = []

        for j in range(numberOfRows):

            row.append(random.randint(-100,100))



        matrix.append(row)

        

    return np.array(matrix)



# 2. Create function to calculate to get minor for a specify value

def minor(matrix, i, j):

    # Get temp matrix in order to calculate minors

    temp = []

    for x in range(3):

        row = []

        for y in range(3):

            if x != i and y != j:

                row.append(matrix[x][y])

                

        if len(row) > 0:

            temp.append(row)

            

    return temp[0][0] * temp[1][1] - temp[0][1] * temp[1][0]       



# 3. Calculate det()

def det(matrix, numberOfRows):

    det = 0

    for i in range(numberOfRows):

        m = minor(matrix, 0, i)

        if i % 2 == 0:

            det += m * matrix[0][i]

        else:

            det -= m * matrix[0][i]

            

    return det

            

# 4. Perform

matrix = createSquareMatrix(3)

determinant = det(matrix, 3)

print("Matrix 3x3: \n", matrix)

print("Use built-in function: ", np.linalg.det(matrix))

print("Final det: ", determinant)
import numpy as np

import random



def createSquareMatrix(numberOfRows):

    matrix = []

    for i in range(numberOfRows):

        row = []

        for j in range(numberOfRows):

            row.append(random.randint(-10,10))



        matrix.append(row)

        

    return np.array(matrix)





def transpose(ma_source, numberOfRows):

    trans = []

    for i in range(numberOfRows):

        row = []

        for j in range(numberOfRows):

            row.append(ma_source[j][i])

            

        trans.append(row)

    return np.array(trans)







def minor(matrix, i, j):

    # Get temp matrix in order to calculate minors

    temp = []

    for x in range(3):

        row = []

        for y in range(3):

            if x != i and y != j:

                row.append(matrix[x][y])

                

        if len(row) > 0:

            temp.append(row)

            

    return temp[0][0] * temp[1][1] - temp[0][1] * temp[1][0]   





def det(matrix, numberOfRows):

    det = 0

    for i in range(numberOfRows):

        m = minor(matrix, 0, i)

        if i % 2 == 0:

            det += m * matrix[0][i]

        else:

            det -= m * matrix[0][i]

            

    return det





def createAdjMatrixFrom(ma_source, numberOfRows):

    adj = []

    for i in range(numberOfRows):

        row = []

        for j in range(numberOfRows):

            value = minor(ma_source, i, j) * pow(-1, i + j)

            row.append(value)

            

        adj.append(row)

        

    return np.array(adj)





def inverse(ma_source, numberOfRows):

    det_matrix = det(ma_source, numberOfRows)

    

    print("\nDet = ", det_matrix)

    

    trans_matrix = transpose(ma_source, numberOfRows)

    print("\nTranspose\n", trans_matrix)

    

    adj_matrix = createAdjMatrixFrom(trans_matrix, numberOfRows)

    

    print("\nAdj:\n", adj_matrix)

    

    invs = []

    for i in range(numberOfRows):

        row = []

        for j in range(numberOfRows):

            row.append(adj_matrix[i][j] * (1 / det_matrix))

                       

        invs.append(row)

        

    return np.array(invs)





# Create matrix

matrix = createSquareMatrix(3)

#matrix = np.array([[1,2,3], [0,1,4], [5,6,0]])

print("Matrix 3x3:\n", matrix)



inv_matrix = inverse(matrix, 3)

print("\nMatrix -1\n", inv_matrix)

print("\nUse built-in function:\n", np.linalg.inv(matrix))
