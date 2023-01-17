#need to create a vector

#import numpy library

import numpy as np
#create a vector row

vector_row=np.array([1,2,3])
vector_row
#create a vector column

vector_col=np.array([[1]

         ,[2]

         ,[3]

         ])
vector_col
#create matrix using 2d array

matrix=np.array([[1,2,3],

              [4,5,6]

             ])
matrix
#create a matrix using matrix object

matrix_object=np.mat([

                [1,2],

                [3,4]

                    ])
matrix_object
#checking the type

type(matrix_object)
#creating spare matrix

#If the data has more zeros than sparse matrix is the best way

from scipy import sparse

matrix=np.array([[1,0,2,0],

                 [1,0,0,1],

                 [0,0,1,0]

                ])

#sparse matrix

matrix_sparse=sparse.csr_matrix(matrix)
print(matrix_sparse)
#selecting value from matrix

matrix=np.array([

    [1,2,3],

    [1,4,5]

])
#selecting 2nd row and 3rd column

matrix[1,2]
#selecting 1st row and 1st column

matrix[0,0]
#selecting 2nd row and 2nd column

matrix[1,1]
vector_row=np.array([1,2,3])
vector_row
#selecting 1st element in vector

vector_row[0]
#selecting all the elements

vector_row[:]
#selecting element till index 2

vector_row[:3]
#selecting last element in the vector

vector_row[-1]
#selecting 2 rows and all the columns

matrix[:2,:]
#selecting 1 row and all column

matrix[:1,:]
#selecting all rows and all the column

matrix[:,:]
#selecting all the row and 1 column

matrix[:1]
#selecting all the rows and 2nd column

matrix[:,1:2]
#selecting all the rows and 1st columns

matrix[:,:1]
#selecting all the element in the second column

matrix[:,2]
#selecting all the element in the 1st column and second column

matrix[:,:2]
matrix=np.array([

    [1,2,3,4],

    [5,6,7,8],

    [9,10,11,12]

])
matrix
#finding the number of rows and columns

matrix.shape
#finding number of element

matrix.size
#finding dimension

matrix.ndim
matrix3d=np.array([

   [ [2,3,4,5],

    [1,2,3,4],

    [2,3,4,5],

    [3,4,3,2]]

])
matrix3d.ndim
matrix3d[0][1][2]
#applying some function to multiple elements

matrix
#lamda is important function that avoids looping over 

add_10=lambda i:i+10
#vectorize method apply some operations to multiple element

vectorized=np.vectorize(add_10)
vectorized(matrix)
matrix
matrix+10
#finding max and min value in matrix

matrix.min()
matrix.max()
#finding max and min in row and column vise

#for row axis=0

print(np.max(matrix,axis=0))

print(np.min(matrix,axis=0))
#column vise axis=1

print(np.min(matrix,axis=1))

print(np.max(matrix,axis=1))
#finding averaqe,variance and standard deviation

matrix
np.mean(matrix)
#sum of difference b/w mean value

np.var(matrix)
#average of difference b/w mean value

np.std(matrix)
#row vise mean

np.mean(matrix,axis=1)
#column vise mean

np.mean(matrix,axis=0)                                                          
matrix.size
#reshaping array

#number of element should be equal in both

matrix.reshape(6,2)
# -1 refer to as many as needed

matrix.reshape(1,-1)
matrix.reshape(-1,1)
matrix
#transposing a matrix

matrix.T
vector_row
#we cannot transpose a row vector

vector_row.T
#we can transpose to a column vector

np.array([[1,2,2]]).T
#flattening a matrix

matrix
matrix.flatten()
#works the same way

matrix.reshape(1,-1)

mat=np.array([

    [1,2,3],

    [4,5,6],

    [5,6,7]

])
#findin rank of a matrix

np.linalg.matrix_rank(mat)
#findind determent of a matrix

np.linalg.det(mat)
#getting diagonal of a matrix

mat.diagonal()
mat
mat.diagonal(offset=2)
mat.diagonal(offset=1)
mat.diagonal(offset=-1)
mat.diagonal(offset=-2)
#finding trace of a matrix

mat.trace()
#same way works

sum(mat.diagonal())
#finding eigencalues and vectors

mat
eigvalues,eigvectors=np.linalg.eig(mat)
eigvalues
eigvectors
#performing mathematical operations

matrix1=np.array([

    [1,1],

    [1,2]

                 ])
matrix2=np.array([

    [2,5],

    [3,7]

])
matrix2.shape
matrix1.shape
np.dot(matrix1,matrix2)
np.add(matrix1,matrix2)
np.subtract(matrix1,matrix2)
#inverting a matrix

np.linalg.inv(matrix1)
#geneating random number

np.random.random(3)
np.random.random(3)
#using seed 

np.random.seed(2)
np.random.random(3)
np.random.random(3)
#using seed we can predict the values



#generating number with various distribution

np.random.randint(0,11,3)
np.random.normal(0.0,0.1,3)
np.random.logistic(0.0,0.1,3)
np.random.uniform(1.0,2.0,3)