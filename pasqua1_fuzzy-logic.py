import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
weights_of_experts = np.array([0.2,

        0.15,

        0.1,

        0.08,

        0.07,

        0.07,

        0.07,

        0.07,

        0.07,

        0.07,

        0.05])

weights_of_experts
dict = {'DenisI' : np.array([

            [0.8, 0.6, 0.5],

            [1, 0.8, 0.7],

            [1, 1, 0.3]]

        ),

        'Sonya' : np.array([

            [0.7, 0.5, 0.2],

            [0.8, 0.8, 0.6],

            [0.9, 0.8, 0.5]]

        ),

        'Maxim' : np.array([

            [0.7, 0.5, 0.2],

            [0.8, 0.8, 0.6],

            [0.9, 0.8, 0.5]]

        ),

        'DimaJ' : np.array([

            [0.7, 0.5, 0.2],

            [0.8, 0.8, 0.6],

            [0.9, 0.8, 0.5]]

        ),

        'SashaV' : np.array([

            [1, 1, 1],

            [0.7, 0.7, 0.8],

            [1, 0.8, 1]]

        ),

        'Alina' : np.array([

            [0.9, 0.9, 0.5],

            [0.8, 0.8, 0.6],

            [0.9, 0.8, 0.5]]

        ),

        'Dasha' : np.array([

            [0.8, 0.4, 0.6],

            [0.8, 0.8, 0.6],

            [0.6, 0.8, 0.4]]

        ),

        'Gleb' : np.array([

            [0.7, 0.5, 0.2],

            [0.8, 0.8, 0.6],

            [0.9, 0.8, 0.5]]

        ),

        'DenisA' : np.array([

            [0.7, 0.2, 0.8],

            [0.8, 0.8, 0.8],

            [0.7, 0.9, 0.8]]

        ),

        'SashaK' : np.array([

            [0.7, 0.5, 0.2],

            [0.8, 0.8, 0.6],

            [0.9, 0.8, 0.5]]

        ),

        'DimaV' : np.array([

            [0.7, 0.5, 0.2],

            [0.8, 0.8, 0.6],

            [0.9, 0.8, 0.5]]

        )

       }

row = col =3

def emptyDoubleArray(): 

    return np.zeros((row, col))

 

def getColumn(A,col):

    return A[:,col]

 

def getRow(A,row):

    return A[row,:]

    

def add(A, other):

    return A+other

        

def mul (A, number): 

    return A*number

 

def div(A,other):

    return A-other

    

def mul_t(A,other):

    result = emptyDoubleArray();

    k = 3

    for i in range (row):

        for j in range (col):

            for k in range (3):

                result[i][j] += A[i][k] * other[k][j]

    return result

 

def toString(A):

    return A.ravel()
def calcVector(A):

    yourWeightParam = np.array([

        0.35,

        0.3,

        0.35]

    )

 

    vector = np.zeros(3)

    col = np.zeros(3)

 

    k = 0

    for i in range (3):

        k = 0

        col = getColumn(A,i)

        for it in range(yourWeightParam.shape[0]):

            vector[i] = vector[i] + col[it] * yourWeightParam[it]

    return vector
def main():

    matrix = np.zeros((3,3))

    i = 0

    

    for k, v in dict.items():

        matrix = matrix + v * weights_of_experts[i]

        i = i + 1

 

    vector = toString(calcVector(matrix))

 

    print(matrix)

    print(vector)

main()