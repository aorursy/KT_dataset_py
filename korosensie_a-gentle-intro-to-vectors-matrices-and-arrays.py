#Problem---Create a vector

#Solution---We are going to create a 1-D(one dimensional) array



import numpy as np #loading the numpy library as np (np is the most common alias used for numpy)  

vector_row = np.array([1,2,3]) #creating a vector as a row



#creating vector as a column ,the number are in nested list i.e. list within list

vector_column=np.array([[1],

                      [2],

                      [3]])

print(vector_row) #printing the vector row

print("\n",vector_column)  #printing the vector column with extra new line
#Problem---Create a Matrix

#solution--- We are going to Use Numpy to create a 2-D array



#import numpy as np (we have already imported the numpy library in previous code so we are using comment to show)

matrix=np.array([[1,2],

               [3,4],

               [5,6]])



#the matrix contains 3 Rows and 2 Columns
print(matrix)
#import numpy as np

from scipy import sparse   # import the sparse library



#creating the matrix



matrix=np.array([[0,0],

                [0,1],

                [2,0]])



# creating the compressed sparse row(CSR) matrix



matrix_sparse =sparse.csr_matrix(matrix)

print(matrix_sparse)
#creating large matrix

matrix_large=np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

                       [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],

                       [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

sparse_matrix_large=sparse.csr_matrix(matrix_large)
print(sparse_matrix_large)
#Problem---Select elements From vectors and matrices

import numpy as np # import or load the numpy library

#create a row vector

vector=np.array([1,2,3,4,5,6,7,8,9])



#creating a matrix 3*3

matrix=np.array([[1, 2, 3],

                 [4, 5, 6],

                 [7, 8, 9]])
#Selecting the 5th element in the vector

vector[4]
#Selecting the 2nd row and 3rd column from matrix

matrix[1,2]
#selecting all the elements of a vector using slicing

vector[:]
#selecting all the elements of a matrix using slicing

matrix[:]
#selecting everything up to and including 3rd element in vector

vector[:3]
#selecting everything after 3rd element in vector

vector[3:]
#selecting last element from the vector

vector[-1]
#selecting the first two rows and all columns of the matrix

matrix[:2,:]
#selecting all rows and the second column of the matrix

matrix[:,1:2]
#Problem---Describe the shape size and dimensions of the matrix

#solution--We will use shape,size,and ndim of numpy

#load the numpy library

import numpy as np



# Creating the matrix

matrix=np.array([[1, 2, 3, 4],

                 [5, 6, 7, 8],

                 [9, 10, 11, 12]])

#Describing the number of rows and columns

matrix.shape
#Describing the number of elements in matrix

matrix.size
#Describing the dimensions of the matrix

matrix.ndim
import numpy as np # importing the numpy library



#creating a matrix

matrix=np.array([[1, 2, 3],

                 [4, 5, 6],

                 [7, 8, 9]])



#creating a lambda function that adds 100 to something

add_100 =lambda i:i+100



#creating a vectorized function

vectorized_add_100=np.vectorize(add_100)



#applying function to all elements in a matrix

vectorized_add_100(matrix)

#adding 100 using broadcasting

matrix1=np.array([[1, 2, 3],

                  [4, 5, 6],

                  [7, 8, 9]])

matrix1 +100
#Problem---Find the max and min value of the array

#Solution-- we will use numpy's min and max function



#import the numpy library

import numpy as np



#creating matrix

matrix=np.array([[1,2,3],

                [4,5,6],

                [7,8,9]])



#returning maximum value or element

np.max(matrix)
#returing minimum value or element

np.min(matrix)
#returing the maximum element in each column

np.max(matrix,axis=0)
#returing the minimum element in each column

np.min(matrix,axis=0)
#returing the maximum element in each row

np.max(matrix,axis=1)
#returing the minimum element in each row

np.min(matrix,axis=1)
#Problem---calculate the descriptive statistics about an array

#solution---we will use mean, var, and std



#import the numpy library

import numpy as np



#creating a matrix

matrix=np.array([[1,2,3],

                [4,5,6],

                [7,8,9]])

#returning the mean

np.mean(matrix)
#returning the variance

np.var(matrix)
#returning the standard deviation

np.std(matrix)
#we can perfom mean standard deviation and variance alongs specific rows and colums just like the minimum and maximum function 

#returns the mean of each rows

np.mean(matrix,axis=0)
#Problem---To change the shape of the array without chnaging the elements value

#Solution---We will use reshape



#import numpy library

import numpy as np



#creating the matrix of shape 4*3

matrix=np.array([[1,2,3],

                [4,5,6],

                [7,8,9],

                [10,11,12]])
#reshaping the matrix in 2*6 matrix

matrix.reshape(2,6)
#one of the useful arguements of reshape is -1,which means "as many as needed" so reshape(1,-1) means one row and as many as column

matrix.reshape(1,-1)

# This means 1 column as many as rows

matrix.reshape(-1,1)
#If we provide single digit it will reshape the array in 1D array

matrix.reshape(12)
#Problem---Transpose a Vector or Matrix

#solution---We will use T method



#import the numpy library

import numpy as np

matrix=np.array([[1,2,3],

                [4,5,6],

                [7,8,9]])
#Transposing a Matrix

matrix.T
#transpose a vector

np.array([1,2,3,4,5,6,7]).T



#The output will be same
#transposing a row vector

np.array([[1,2,3,4,5,6,7]]).T

#Problem---Transform the matrix in 1-D aray

#solution---We will use flatten



#import numpy library

import numpy as np



#creating matrix

matrix=np.array([[1,2,3],

                [4,5,6],

                [7,8,9]])



#flatten matrix

matrix.flatten()
#falttening using reshape()

matrix.reshape(1,-1)
#Problem---Know the rank of the Matrix

#Solution---we will use matrix_rank 



#import the numpy library

import numpy as np



#Creating the matrix

matrix=np.array([[1,2,3],

                [3,6,9],

                [4,5,7]])
#returning the rank of matrix

np.linalg.matrix_rank(matrix)
#Problem---Find the determinant of the matrix

#Solution---We will use linearalgebra det method



#import numpy library

import numpy as np



#creating a matrix



matrix=np.array([[1, 2, 3],

                 [2, 4, 6],

                 [3, 8, 9]])
#Returning the determinant of the matrix

np.linalg.det(matrix)
#Problem---get the diagonal of the matrix

#Solution---We will use diagonal method



#import numpy library

import numpy as np



#Creating a matrix

matrix=np.array([[1, 2, 3],

                 [2, 4, 6],

                 [3, 8, 9]])
#returning the diagonal of the matrix

matrix.diagonal()
#We can get the subset of the diagonal by using offset argument

#Returing the diagonal one above the main diagonal

matrix.diagonal(offset=1)
#returing the diagonal two above the main diagonal

matrix.diagonal(offset=2)
#returing the diagonal one below the main diagonal

matrix.diagonal(offset=-1)
#Problem---Calculate the trace of  the given matrix

#Solution---We will use the trace method



#importing the numpy library

import numpy as np



#creating a matrix

matrix=np.array([[1, 2, 3],

                 [2, 4, 6],

                 [3, 8, 9]])
#Returning Trace

matrix.trace()
#Return diagonal and sum elements

sum(matrix.diagonal())
#Problem---Find the Eigenvalues and the eigenvectors of a square Matrix

#Solution--- We will use linearalgebra's eig



#import numpy library

import numpy as np



#creating a matrix

atrix=np.array([[1, 2, 3],

                 [2, 4, 6],

                 [3, 8, 9]])
#calculte eigenvalues and eigenvectors

eigenvalues,eigenvectors=np.linalg.eig(matrix)
#Returning eigenvalues

eigenvalues
#Returning eigenvectors

eigenvectors
#Problem---Calculate the dot product of two vectors

#Solution---We will use numpy's dot method



#import numpy library

import numpy as np



#creating two vectors

vector_a = np.array([1,2,3])

vector_b = np.array([4,5,6])



#calculating dot product

np.dot(vector_a,vector_b)
#In Python3 we can also use @ operator to calculate the dot product

vector_a@vector_b
#Problem---Add,subtract and multiply two matrices

#Solution---We will use add,subtract and dot 



#importing numpy library

import numpy as np



#creating two matrix

matrix_a = np.array([[1, 1, 1],

                     [1, 1, 1],

                     [1, 1, 2]])



matrix_b = np.array([[1, 3, 1],

                     [1, 3, 1],

                     [1, 3, 8]])
#Adding two matrices

np.add(matrix_a,matrix_b)
#Alternate way to add two Matrices

matrix_a+matrix_b
#Subtracting two matrices

np.subtract(matrix_a,matrix_b)
#Alternate way to subtarct two matrices

matrix_a-matrix_b
#Multiplying two matrices using numpy's dot

np.dot(matrix_a,matrix_b)
#alternate way of multiplying using @ python operator

matrix_a@matrix_b
#If we want to do element wise multiplication we can use the * operator

matrix_a*matrix_b
#Problem---Find the inverse of the square Matrix

#Solution---we will use linear algebra inv method 



#importing numpy lib

import numpy as np



#Creating a matrix

matrix=np.array([[1,2],

                [3,4]])
#calculating the inverse of the matrix

np.linalg.inv(matrix)
#Problem---Generate pseudorandom values

#Solution--- We will use numpy's random



#importing numpy library

import numpy as np



#seed is used to get predicatble,repeatable results

#setting seed

np.random.seed(0)



#generating 3 random float number between 0.0 and 1.0

np.random.random(5)
#Generating 5 random integers between 1 and 100

np.random.randint(0,101,5)
#Generating 5 numbers from a normal distribution where mean is 0.0 and standard deviation is 1.0

np.random.normal(0.0,1.0,5)
#generating 5 numbers from a logistic distribution where mean is 0.0 and standard deviation is 1.0

np.random.logistic(0.0,1.0,5)
#generating 5 numbers greater than 5.0 and less than 10.0

np.random.uniform(5.0,10.0,5)