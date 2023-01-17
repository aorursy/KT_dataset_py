# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# (scipy) library for more advanced linear algebra and factorial of a number
from scipy import linalg, sparse, misc

# polyld from numpy allows the programmer to create polynomial objects while array is used to creat an array and tensordot finds the dot product of tensors
from numpy import poly1d, array, tensordot


#creates a polynomial object (binomial expansion)
poly= poly1d([1,1,1,1,1], [1,1,1],[1,1,1])
print(poly)

# 2 variable table
n1=([(2,4+1,5) , (5,6,7)])
print(n1)

# a different type of 2 variable table (maybe for pixels, images and sound)
n2=([(2j,4j+1,5j) , (5j,6j,7j)])
print(n2)

#storing as 2 different matrixes
c = np.array([[(1.5,2,3), (4,5,6)], [(3,2,1), (4,5,6)]])
print(c)

# y is a 9*9 matrix
y = np.array([[1,2,3,3,4,5,3,4,5]
              ,[2,4,5,2,4,5,2,4,5]
            ,[1,2,3,3,4,5,3,4,5]
              ,[2,4,5,3,4,5,2,4,5]
             ,[1,2,3,3,4,5,3,4,5]
              ,[2,4,5,2,4,5,2,4,5]
             ,[1,2,3,3,4,5,3,4,5]
              ,[2,4,5,2,4,5,2,4,5]
             ,[1,2,3,3,4,5,3,4,5]])

# x is a 9*2 matrix
x = np.array([[1,2],[3,4],[5,6],[7,8],[9,9],[3,4],[5,6],[7,8],[9,9]])

# finds the dot product of matrix y and matrix x
print(np.dot(y,x))

#factorial of 5
print(misc.factorial(5))

a = np.array(
[[2,3],
 [2,3]])
b = np.array(
[[3,4],
 [5,6]])

#flattens the b array (matrix) into a one dimensional matrix
print(b.flatten())


#add 2 matrices
print(a+b)

#subtract 2 matrices
print(a-b)

print(np.dot(a,b))

# basix multiplication of matix a with matrix b
print(a*b)

# T stands for transposing the matrix b
print(b.T)

print(y.T)

# shape is used to find how many rows and colums are in matrix a 
print(a.shape)

xx=np.dot(y,x)
print(xx.shape)


#finding the inverse of matrix b *works only when each row and column are different and when all elements are non-zero
print(linalg.inv(b))

#finding the determinant of b *works only on square matrices 
print(linalg.det(b))

print(linalg.det(y))

# In order to obtain each multiplication of at a dense mesh grid consisting of 2 different multi-dimensional matrixes
[q,p]= np.mgrid[0:5,0:5]
print(q)
print(p)
print(q*p)

#creates an open mesh grid of 2 differnt one dimensional matrixes
np.ogrid[0:5, 0:2]

#finding the solution for a simultaneous equations with a dense matrix (matrix with more non-zero elements) (while sparze matrices contain more zero elements) 
SimultaneousEquations=([[2, 1], [3, 2]])
bAnswers=([[1],[4]])
solution= linalg.solve(SimultaneousEquations, bAnswers)
print(solution)

#finding the frobenius norm of a matrix 
Matrix1=([[2, 5], [8, 2]])
print(linalg.norm(Matrix1))

#finding the tensor product of 2 tensors when axes=0 (axes takes in values 1 and 2 only) (with 2 basis vectors)
tensor1=array([2,3])
tensor2=array([4,5])
print(np.tensordot(tensor1, tensor2, axes=0))

#finding the tensor product of 2 tensors when axes=1 (axes takes in values 1 and 2 only) (with 3 basis vectors)
tensor3=array([3,3,3])
tensor4=array([4,5,4])
print(np.tensordot(tensor3, tensor4, axes=1))

#finding the tensor product of 2 tensors when axes=0 (axes takes in values 1 and 2 only) (with 3 basis vectors)
tensor1=array([2,3,3])
tensor2=array([4,5,3])
print(np.tensordot(tensor1, tensor2, axes=0))

#finding the tensor product of 2 tensors when axes=0 (axes takes in values 1 and 2 only) (with 2 basis vectors)
tensor3=array([3,3])
tensor4=array([4,5])
print(np.tensordot(tensor3, tensor4, axes=1))

#finding the kronecker product of matrices kp1 and kp2 ( used especially in econometrics -the application of statistical methods to economic data)
kp1=([[1,2],[3,4]])
kp2=([[2,4],[5,3]])
print(np.kron(kp1,kp2))

#finding the matrix rank
print(np.linalg.matrix_rank(kp1))

# finding the pseudo-inverse of matrix kp1 using the least squared method (used to find a linear model-in graphs, and when a system is overdetermined)
print(linalg.pinv(kp1))

#finding the exponential of a matrix - used to solve linear differential equations *go to http://www.thefullwiki.org/Matrix_exponential
print(linalg.expm(kp1))

#finding the logarithm of a matrix * got to https://en.wikipedia.org/wiki/Logarithm_of_a_matrix AND https://math.stackexchange.com/questions/178929/matrix-logarithms
print(linalg.logm(kp1))

#finding the Matrix sine
print(linalg.sinm(kp1))

#finding the Matrix cosine
print(linalg.cosm(kp1))

#finding the Matrix tangent
print(linalg.tanm(kp1))

#matrix square root
print(linalg.sqrtm(kp1))

#arbitrary functions: evaluate matrix function *implements the general algorithm based on Schur decomposition *go to https://en.wikipedia.org/wiki/Schur_decomposition
print(linalg.funm(kp1, lambda x: x*x))



#unpacks eigenvalues of matrix kp1
print(linalg.eigvals(kp1))

#eigenvalues and eigenvectors have applications in Schrodinger's equation, eigenfaces(in image processing), molecular orbitals, stress tensor, moment of inertia tensor
#eigenvalues and eigenvectors also apply in vibration analysis, basic reproduction number(the fundamental number in the study how infectious disease spread), geometric transformations
#eigenvalues and eigenvectors also have applications in google's pagerank algorithm, principal component analysis(in bioinformatics, data mining, psychology and marketing)
la, v = linalg.eig(kp1)  #Solve ordinary or generalized eigenvalue problem for a square matrix
l1, l2 = la #unpack eigenvalues
print(la)
print(v[:,0])  #first eigenvector 
print(v[:,1])  #second eigenvector
print(linalg.eigvals(kp1)) #unpacks eigenvalues








# Any results you write to the current directory are saved as output.
