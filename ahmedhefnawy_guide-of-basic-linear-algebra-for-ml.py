# scipy
import scipy
print('scipy: %s' % scipy.__version__)
# numpy
import numpy
print('numpy: %s' % numpy.__version__)
# matplotlib
import matplotlib
print('matplotlib: %s' % matplotlib.__version__)
# pandas
import pandas
print('pandas: %s' % pandas.__version__)
import numpy as np
# create normal array__________________
l = [1.0, 2.0, 3.0]
a = np.array(l)
# display array
print(a)
# display array shape
print(a.shape)
# display array data type
print(a.dtype)
# create empty array ________________
a = np.empty([3,5])
print(a)
# create zero array
a = np.zeros([4,5])
print(a)
# create one array
a = np.ones([4,5])
print(a)
np.arange(2,10,2) # (start , stop , Step)
# create array with vstack ###################
# create first array
a1 = np.array([1,2,3])
print("_______arr 1______")
print(a1)
# create second array
a2 = np.array([4,5,6])
print("_______arr 2______")
print(a2)
# vertical stack
a3 = np.vstack((a1, a2))
print("\n++++---- arr 1 >> vertical stack >> arr 2 ----++++")
print(a3)
print("\n-------- SHAP >>>>")
print(a3.shape)
# create array with hstack
# create first array
a1 = np.array([1,2,3])
print("_______arr 1______")
print(a1)
# create second array
a2 = np.array([4,5,6])
print("_______arr 2______")
print(a2)
# create horizontal stack
a3 = np.hstack((a1, a2))
print("______arr 1 >> HORIZONTAL stack >> arr 2_______")
print(a3)
print("\n-------- SHAP >>>>")
print(a3.shape)
# create one-dimensional array
# list of data
data = [11, 22, 33, 44, 55]
# array of data
data = np.array(data)
print(data)
print(type(data))
# create two-dimensional array
# list of data
data = [[11, 22],
[33, 44],
[55, 66]]
# array of data
data = np.array(data)
print(data)
print(type(data))
# define array
data = np.array([11, 22, 33, 44, 55])
# index data
print(data[0])
print(data[4])
#print(data[5])# Access Error Wrong loocaion
print(data[-1])
print(data[-5])
# index two-dimensional array
""" 
    in c/c++  >> data[0,0]
    in python >> data[0][0]
"""
# define array
data = np.array([
[11, 22],
[33, 44],
[55, 66]])
# index data
print(data[0,0])
print(data[0,]) # [ROW , Coulmn]
# define array
data = np.array([11, 22, 33, 44, 55])
print(data[0:1])
print(data[0:2])
print(data[0:3])
print(data[0:4])
print(data[0:5])
print(data[:])
print("_____________________\n")
#print(data[5:5])
print(data[4:5])
print(data[3:5])
print(data[2:5])
print(data[1:5])
# negative slicing of a one-dimensional array
print(data[-2:])
# split input and output data
# define array
data = np.array([
[11, 22, 33],
[44, 55, 66],
[77, 88, 99]])
# separate data
X, y = data[ : , : -1], data[ : , -1]
print("_______ ALL DATA_____________\n")
print(data)
print("\n_______data[ : , : -1]_________\n")
print(X)
print("\n_______data[ : , -1]_________\n")
print(y)
print("=================\n")
s = data[ -1 ,  : ]
print("s >>> \n" , s)
data = np.array([
[11, 22, 33],
[44, 55, 66],
[77, 88, 99]])
# separate data
X, y = data[ : , : -1 ] , data[ : , -1 ]
print(X)
print("____________")
print(y)
data = np.array([
[11, 22, 33],
[44, 55, 66],
[77, 88, 99],
[10, 10, 10]])
# print real data
print("The REAL Data ---\n" , data , '\n')
# initializing split Value
split = 2 
print("Split value = " , split , "\n")
# Splitting srtep - separate data -
train,test = data[ : split , : ] , data [ split : , : ]
# Printing 
print("TrainSet >> data[ : split , : ]: \n",train)
print("\nTestSet >> data [ split : , : ] : \n",test)
""" one-dimensional array """
# define array
data = array([1, 2, 3, 4, 5 , 6])
# shape
print(data.shape)
""" two-dimensional array """
# list of data
data = [[11, 22],
[33, 44],
[55, 66]]

# array of data
data = array(data)
print(data.shape)
# row and column shape of two-dimensional array
# list of data
data = [[11, 22],
        [33, 44],
        [55, 66]]

#array of data
data = array(data)

print('Number of Rows: {}'.format( data.shape[0]) ) # zero for ROW
print('Number of Columnss: {}'.format(data.shape[1])) # one for Column
np.arange(2,10,2)#.reshape(2,5)
x = np.arange(10).reshape(2,5)
x
# define array
data = np.array([11, 22, 33, 44, 55])
print( "Original Array \n" , data ,'\n')
print("||||| original shap :---->",data.shape)
# reshape
""" reshape ( Rows , Columns ) """
data = data.reshape(5, 1) 
print( "=================== \n Reshaped Array \n" , data ,'\n')
print("||||| Updated Shap :---->",data.shape)

#np.reshape()
# list of data
data = [[11, 22 , 12 ,0],
[33, 44 , 34 ,0],
[55, 66 , 56 ,0],
[33, 44 , 34 ,0]]

# array of data
data = np.array(data)
print("Original Data \n", data)
print("\n Data shap |-->",data.shape,'\n______________________________')

# reshape
data_R = data.reshape( 4 , 4 , 1)
#++++++++++++++++++++
print("Reshaped Array ||--->\n",data_R)
print("\n New Data shap |-->", data_R.shape ,'\n')
# broadcast scalar to one-dimensional array
# define array
a = np.array([1, 2, 3])

# define scalar
b_Scalar = 2

# broadcast
c = a + b_Scalar

print("Original Array \n", a ,'\n' )
print("----------\nScalar Value = ",b_Scalar,'\nbroadcast -> c = a + b \n------------')
print("\nAfter Broadcasting\n",c)
# broadcast scalar to two-dimensional array

# define array
A = np.array([
[1, 2, 3],
[1, 2, 3]])

# define scalar
b_Scalar = 2

# broadcast
C = A + b_Scalar

print("Original Array \n", A ,'\n' )
print("----------\nScalar Value = ",b_Scalar,'\nbroadcast -> C = A + b \n------------')
print("\nAfter Broadcasting\n",C)
# broadcast one-dimensional array to two-dimensional array
# define two-dimensional array
A = array([
[1, 2, 3],
[1, 2, 3]])

# define one-dimensional array
b_Scalar = array([1,2,3])

# broadcast
C = A + b_Scalar

print("Original Array \n", A ,'\n' )
print("----------\nScalar Value = ",b_Scalar,'\nbroadcast -> C = A + b \n------------')
print("\nAfter Broadcasting\n",C)
# broadcast one-dimensional array to two-dimensional array
# define two-dimensional array
A = np.array([
[1, 2, 3],
[1, 2, 3]])

# define one-dimensional array
b_Scalar = np.array([1,2])

print('The shap of Array :' , A.shape ,'\n')
print('The shap of Scalar :' , b_Scalar.shape ,'\n')
print('!!ERROR broadcast >>\n','ValueError: operands could not be broadcast together with shapes (2,3) (2,) \n \n')

# !!ERROR broadcast
C = A + b_Scalar
print("Original Array \n", A ,'\n' )
print("----------\nScalar Value = ",b_Scalar,'\nbroadcast -> C = A + b \n------------')
#print("\nAfter Broadcasting\n",C)
# >>>>>>>>>>>>>>>>>> create a vector
# define vector
v = np.array([1, 2, 3])
print('Vector V =',v)
# >>>>>>>>>>>>>>>>> vector addition
# define first vector
a = np.array([1, 2, 3])# V1
print("First-Vector",a)
# define second vector 
b = np.array([1, 2, 3]) # V2
print("Second-Vector",b)
# add vectors
c = a + b
print('Operation : c = a + b')
print("\nResult Vector",c)
# vector subtraction
# define first vector
a = np.array([1, 2, 3])

print('First-Vector :',a)
# define second vector
b = np.array([0.5, 0.5, 0.5])

print('Second-Vector',b)
# subtract vectors
c = a - b
print('Operation : c = a - b')
print('\nResult-Vector :',c)
# vector multiplication
from numpy import array
# define first vector
a = np.array([1, 2, 3])

# define second vector
b = np.array([1, 2, 3])
# multiply vectors
c = a * b
print('First-Array',a)
print('Second-Array',b)
print('Operation : c = a * b')
print('\nResult-Array',c)
# vector division
# define first vector
a = np.array([1, 2, 3])

# define second vector
b = np.array([1, 2, 3])

# divide vectors
c = a / b

print('First-Array',a)
print('Second-Array',b)
print('Operation : C = a/b')
print('\nResult-Array',c)
# vector dot product
# define first vector
a = np.array([1, 2, 3])
# define second vector
b = np.array([1, 2, 3])
# multiply vectors
c = a.dot(b)
print('First-Vector :',a)
print('Second-Vector :',b)
print('______________\nmultiply Operation' , a*b)
print('Result Dot Product :',c)
#>>>>>>>> vector-scalar multiplication
# define vector
a = np.array([1, 2, 3])
b = np.array([1, 2, 3])

# define scalar
scala_R = 0.5

# multiplication
c = scala_R * a

print('Vectoer :',a)
print('Scalar :',scala_R)
print('\nResult Dot Product with b Vector :',a.dot(b))
print('\nVector-Scalar Multiplication',c)

# vector L1 norm
# define vector
from numpy.linalg import norm
a = np.array([1, 2, 3])
print("Vector >",a)

# calculate norm
l1 = norm(a, 1)

print('L1 Norm> ',l1)
# vector L2 norm
# define vector
from numpy.linalg import norm
a = np.array([1, 2, 3])
print('Vector >>',a)

# calculate norm
l2 = norm(a)
print('L2 norm',l2)
# vector max norm
from math import inf
# define vector
a = np.array([1, 2, 3 , 5, 1, 7,  12, 2])
print('Vector >>',a)
# calculate norm
maxnorm = np.norm(a, inf)
print('Max_Norm >>',maxnorm)
# create matrix

A = np.array([[1, 2, 3], [4, 5, 6]])
print('Created Matrix >>\n',A)
# matrix addition
# define first matrix
A = np.array([
[1, 2, 3],
[4, 5, 6]])
# define second matrix
B = np.array([
[1, 2, 3],
[4, 5, 6]])

# add matrices
C = A + B
print('Matrix A :\n', A )
print('\nMatrix B :\n',B )
print('\nOperation : C = A + B \n' )
print('_____________\nResult-Matrix \n',C)
# matrix subtraction
# define first matrix
A = np.array([
[1, 2, 3],
[4, 5, 6]])

# define second matrix
B = np.array([
[0.5, 0.5, 0.5],
[0.5, 0.5, 0.5]])

# subtract matrices
C = A - B
print('Matrix A :\n', A )
print('\nMatrix B :\n',B )
print('\nOperation : C = A - B \n' )
print('_____________\nResult-Matrix \n',C)
# matrix division
# define first matrix
A = np.array([
[1, 2, 3],
[4, 5, 6]])

# define second matrix
B = np.array([
[1, 2, 3],
[4, 5, 6]])

# divide matrices
C = A / B

print('Matrix A :\n', A )
print('\nMatrix B :\n',B )
print('\nOperation : C = A/B \n' )
print('_____________\nResult-Matrix \n',C)
# matrix Hadamard product
from numpy import array
# define first matrix
A = np.array([
[1, 2, 3],
[4, 5, 6]])

# define second matrix
B = np.array([
[1, 2, 3],
[4, 5, 6]])

# multiply matrices
C = A * B


print('Matrix A :\n', A )
print('\nMatrix B :\n',B )
print('\nOperation : C = A * B \n' )
print('_____________\nResult-Matrix \n',C)
# matrix dot product
# define first matrix
A = np.array([
[1, 2],
[3, 4],
[5, 6]])
print('Matrix A :\n',A)
# define second matrix
B = np.array([
[1, 2],
[3, 4]])
print('Matrix B :\n',B)
# multiply matrices
C = A.dot(B)
print('\n-----Result of -multiply matrices :\n',C)
# multiply matrices with @ operator
D = A @ B
print('\n-----Result of -multiply matrices with @ operator :\n',D)

#I recommend using the dot() function for matrix multiplication for now given the newness of the @ operator
# matrix-vector multiplication

# define matrix
A = np.array([
[1, 2],
[3, 4],
[5, 6]])
print('Matrix :\n',A)
# define vector
B = np.array([0.5, 0.5])
print('Vector v :\n',B)
# multiply
C = A.dot(B)
print('\n The result - VECTOR :\n',C)
# matrix-scalar multiplication

# define matrix
A = np.array([[1, 2], [3, 4], [5, 6]])
print('Matrix A : \n',A)
# define scalar
b = 0.5
print('\nSCALAR b =',b)
# multiply
C = A * b
print('________________\nRsult - MATRIX : \n',C)
# triangular matrices
# define square matrix
M = np.array([
[1, 2, 3],
[1, 2, 3],
[1, 2, 3]])
print('Square Matrix :\n',M)

# lower triangular matrix
lower = np.tril(M)
print('\nlower triangular matrix : \n',lower)

# upper triangular matrix
upper = np.triu(M)
print('upper triangular matrix : \n',upper)
# diagonal matrix

# define square matrix
M = np.array([
[1, 2, 3],
[1, 2, 3],
[1, 2, 3]])
print('square matrix : \n',M)
# extract diagonal vector
d = np.diag(M)
print('\n________________________|  extract diagonal vector : \n',d)
# create diagonal matrix from vector
D = np.diag(d)
print('\n________________________|  create diagonal matrix from vector : \n',D)
# identity matrix
I = np.identity(3)
print('Identity Matrix : \n-------------- \n',I)
# orthogonal matrix
from numpy.linalg import inv

# define orthogonal matrix
Q = np.array([
[1, 0],
[0, -1]])
print('orthogonal matrix : \n',Q)
# inverse equivalence
V = inv(Q)

print('inverse of the orthogonal matrix : \n',Q.T)
print('transpose of the orthogonal matrix : \n',V)
# identity equivalence
I = Q.dot(Q.T)
print('\n-----------------------------------------\nidentity matrix is printed which is calculated from the dot product of the orthogonal matrix with its transpose : \n',I)
# transpose matrix
# define matrix
A = np.array([
[1, 2],
[3, 4],
[5, 6]])
print('Matrix A: \n',A)
# calculate transpose
C = A.T
print('A transpose : \n',C)
# invert matrix
from numpy.linalg import inv

# define matrix
A = np.array([
[1.0, 2.0],
[3.0, 4.0]])
print('Matrix A: \n',A)

# invert matrix
B = inv(A)
print('_____________________\n invert matrix A ^-1 : \n',B)
# multiply A and B
I = A.dot(B)
print('______________________ \n\n Identity matrix that calc by multiply A.dot(B) : \n \n',I)
# matrix trace

# define matrix
A = np.array([
[1, 2, 3],
[4, 5, 6],
[7, 8, 9]])

print('Square Matrix : \n',A)

# calculate trace
B = np.trace(A)
print('_______________ \nTrace Matriix : \n',B)
# matrix determinant
from numpy import array
from numpy.linalg import det
# define matrix
A = np.array([
[1, 2],
[4, 5]])

X = np.array([
[1, 2, 3],
[4, 5, 6],
[7, 8, 9]]) 

# calculate determinant
B = det(A)
C = det(X)
print('Matrix A : \n',A)
print('determinant |A| ',B)
print('__________________\nMatrix X : \n',X)
print('determinant |X| ',C)
# vector rank

from numpy.linalg import matrix_rank
# rank
v1 = np.array([1,2,3])
print('Vector-one : \n',v1)
vr1 = matrix_rank(v1)
print('calculating the rank of Vector-One =  \n',vr1)
# zero rank
v2 = np.array([0,0,0,0,0])
print('============================== \nVector-Two :\n',v2)
vr2 = matrix_rank(v2)
print('calculating the rank of Vector-Two =  \n',vr2)
# matrix rank
from numpy.linalg import matrix_rank
# rank 0
M0 = np.array([
[0,0],
[0,0]])
print('Matrix one : \n',M0)
mr0 = matrix_rank(M0)
print('Matrix Rank :: \n',mr0)
# rank 1
M1 = np.array([
[1,2,100,6],
[2,4,90,12]])
print('____________________\n Matrix two : \n',M1)
mr1 = matrix_rank(M1)
print('Matrix Rank : \n',mr1)
# rank 2
M2 = np.array([
[1,2],
[3,4],
[1,2]])
print('____________________ \nMatrix Three : \n',M2)
mr2 = matrix_rank(M2)
print('MAtrix Rank',mr2)
# sparse matrix

from scipy.sparse import csr_matrix
# create dense matrix
A = array([
[1, 0, 0, 1, 0, 0],
[0, 0, 2, 0, 0, 1],
[0, 0, 0, 2, 0, 0]])

print('The dense Matrix ....  \n',A)

# convert to sparse matrix (CSR method)
S = csr_matrix(A)
print('\nconvert to sparse matrix (CSR method) ............ \n', S)
# reconstruct dense matrix
B = S.todense()
print('\nreconstruct dense matrix ______________\n',B)
# sparsity calculation

# create dense matrix
A = np.array([
[1, 0, 0, 1, 0, 0],
[0, 0, 2, 0, 0, 1],
[0, 0, 0, 2, 0, 0]])
print('dense matrix ..........\n',A)

# calculate sparsity
sparsity = 1.0 - count_nonzero(A) / A.size
print('\n__________________ \ncalculate sparsity .......\n',sparsity)
# create tensor

T = np.array([
[[1,2,3], [4,5,6], [7,8,9]],
[[11,12,13], [14,15,16], [17,18,19]],
[[21,22,23], [24,25,26], [27,28,29]]])
print('The shape of the tesor ______ \n',T.shape)
print('\n ---------------------\nTensor 3*3*3 \n',T)
# tensor addition

# define first tensor
A = np.array([
[[1,2,3], [4,5,6], [7,8,9]],
[[11,12,13], [14,15,16], [17,18,19]],
[[21,22,23], [24,25,26], [27,28,29]]])
# define second tensor
B = np.array([
[[1,2,3], [4,5,6], [7,8,9]],
[[11,12,13], [14,15,16], [17,18,19]],
[[21,22,23], [24,25,26], [27,28,29]]])
# add tensors
C = A + B
#print('First tensor \n' , A)
#print('\nSecond tensor \n', B)
print('________________________\nThe result of the Adding operatiopn \n',C)
# tensor subtraction

# define first tensor
A = np.array([
[[1,2,3], [4,5,6], [7,8,9]],
[[11,12,13], [14,15,16], [17,18,19]],
[[21,22,23], [24,25,26], [27,28,29]]])
# define second tensor
B = np.array([
[[1,2,3], [4,5,6], [7,8,9]],
[[11,12,13], [14,15,16], [17,18,19]],
[[21,22,23], [24,25,26], [27,28,29]]])
# subtract tensors
C = A - B
print('The Result of the Subtraction Operation \n',C)
# tensor division
from numpy import array
# define first tensor
A = array([
[[1,2,3], [4,5,6], [7,8,9]],
[[11,12,13], [14,15,16], [17,18,19]],
[[21,22,23], [24,25,26], [27,28,29]]])
# define second tensor
B = array([
[[1,2,3], [4,5,6], [7,8,9]],
[[11,12,13], [14,15,16], [17,18,19]],
[[21,22,23], [24,25,26], [27,28,29]]])
# divide tensors
C = A / B
print('The Resulted Divided tesnsors \n--------------------------------\n ',C)
# tensor Hadamard product

# define first tensor
A = np.array([
[[1,2,3], [4,5,6], [7,8,9]],
[[11,12,13], [14,15,16], [17,18,19]],
[[21,22,23], [24,25,26], [27,28,29]]])
# define second tensor
B = np.array([
[[1,2,3], [4,5,6], [7,8,9]],
[[11,12,13], [14,15,16], [17,18,19]],
[[21,22,23], [24,25,26], [27,28,29]]])
# multiply tensors
C = A * B
print('The result of the multiply tensors (Tensor Hadamard Product) \n',C)
# tensor product

from numpy import tensordot
# define first vector
A = np.array([1,2])
# define second vector
B = np.array([3,4])
# calculate tensor product
C = tensordot(A, B, axes=0)
print('The first Vector \n',A)
print('The first Vector \n',B)
print('\nThe Resulted tensor product \n',C)
# LU decomposition
from scipy.linalg import lu

# define a square matrix
A = array([
[1, 2, 3],
[4, 5, 6],
[7, 8, 9]])
print('The Square Matrix ...........\n',A)

# factorize
P, L, U = lu(A)
# where P is a permutation matrix, L lower triangular with unit
# diagonal elements, and U upper triangular
print('\nP is a permutation matrix : \n',P)
print('\nL lower triangular with unit diagonal elements : \n',L)
print('\n U upper triangular : \n',U)

# reconstruct
B = P.dot(L).dot(U)
print('_________________________________________________\nreconstruct The Decomposition Process \n',B)
# QR decomposition
from numpy.linalg import qr

# define rectangular matrix
A = np.array([
[1, 2],
[3, 4],
[5, 6]])
print('The Original Matrix ...........\n',A)
# factorize
Q, R = qr(A, 'complete')
"""
Signature: qr(a, mode='reduced')
Docstring:
Compute the qr factorization of a matrix.

Factor the matrix `a` as *qr*, where `q` is orthonormal and `r` is
upper-triangular.

Parameters
----------
a : array_like, shape (M, N)
    Matrix to be factored.
    
* 'reduced'  : returns q, r with dimensions (M, K), (K, N) (default)
    * 'complete' : returns q, r with dimensions (M, M), (M, N)
    * 'r'        : returns r only with dimensions (K, N)
    * 'raw'      : returns h, tau with dimensions (N, M), (K,)
    * 'full'     : alias of 'reduced', deprecated
    * 'economic' : returns h from 'raw', deprecated.
"""
print("\nQ >>>>>>>>>>>>>>>> \n",Q)
print('\nR >>>>>>>>>>>>>>>> \n',R)
# reconstruct
B = Q.dot(R)
print('\n_________________________________ \nreconstruct The original matrix \n',B)
# Cholesky decomposition
from numpy import array
from numpy.linalg import cholesky
# define symmetrical matrix
A = array([
[2, 1, 1],
[1, 2, 1],
[1, 1, 2]])
print('symmetrical matrix \n',A)
# factorize
L = cholesky(A)
print('factorize Cholesky decomposition \n',L)
# reconstruct
B = L.dot(L.T)
print('\n_________________________________ \nreconstruct The original matrix \n',B)
# eigendecomposition

from numpy.linalg import eig

# define matrix
A = np.array([
[1, 2, 3],
[4, 5, 6],
[7, 8, 9]])
print('The Matrix A \n' , A)

# factorize
values, vectors = eig(A)

print('\n The EIGEN VALUE______________________________\n\n',values)
print('\n\n The EIGEN VAECTOR_____________________________\n\n',vectors)
# confirm eigenvector

from numpy.linalg import eig
# define matrix
A = np.array([
[1, 2, 3],
[4, 5, 6],
[7, 8, 9]])

# factorize
values, vectors = eig(A)

# confirm first eigenvector
B = A.dot(vectors[:, 0])
print('The Original Matrix \n',A)
print('EIGEN VECTOR " \n' , values ,'\n')
print('______________________multiplies the original matrix with the first eigenvector :  \n',B , '\n')
C = vectors[:, 0] * values[0]
print('______________________compares it to the first eigenvector multiplied by the first eigenvalue \n',C)
# reconstruct matrix
from numpy.linalg import inv

from numpy.linalg import eig
# define matrix
A = np.array([
[1, 2, 3],
[4, 5, 6],
[7, 8, 9]])
print('Real Matrix : \n',A)

# factorize
values, vectors = eig(A)
print('\n Eigen Vectors : \n' , vectors )
# create matrix from eigenvectors
Q = vectors
print('\n matrix from eigenvectors\n' , Q)

# create inverse of eigenvectors matrix
R = inv(Q)
print('\n inverse of eigenvectors matrix \n' , R)

# create diagonal matrix from eigenvalues
L = np.diag(values)
print('\n diagonal matrix from eigenvalues \n' , L)

# reconstruct the original matrix
B = Q.dot(L).dot(R)
print('\n _____________________________\nreconstruct the original matrix \n(  B = (eigenvectors).(eigenvalues).(inverse eigenvectors matrix) ) : \n\n',B)
# singular-value decomposition

from scipy.linalg import svd
# define a matrix
A = np.array([
[1, 2],
[3, 4],
[5, 6]])
print('The Original Matrx : \n',A)

# factorize
U, s, V = svd(A)

print('\nU____________\n',U)
print('\nS____________\n',s)
print('\nV____________\n',V)
# reconstruct rectangular matrix from svd

from scipy.linalg import svd
# define matrix
A = np.array([
[1, 2],
[3, 4],
[5, 6]])
print('__________________________THe Original Matrix : \n',A)

# factorize
U, s, V = svd(A)

# create m x n Sigma matrix
Sigma= np.zeros((A.shape[0], A.shape[1]))
print('\n create m x n Sigma matrix : \n',Sigma)

# populate Sigma with n x n diagonal matrix
Sigma[:A.shape[1], :A.shape[1]] = np.diag(s)
print('\n___populate Sigma with n x n diagonal matrix \n',s)

# reconstruct matrix
B = U.dot(Sigma.dot(V))
print('\n_________________________reconstruct matrix :\n',B)
# reconstruct square matrix from svd

from scipy.linalg import svd
# define matrix
A = np.array([
[1, 2, 3],
[4, 5, 6],
[7, 8, 9]])
print('_______________________________The Original Matrix : \n',A)

# factorize
U, s, V = svd(A)

# create n x n Sigma matrix
Sigma = np.diag(s)

# reconstruct matrix
B = U.dot(Sigma.dot(V))
print('_______________________________The reconstructed matrix : \n',B)
# pseudoinverse

from numpy.linalg import pinv
# define matrix
A = np.array([
[0.1, 0.2],
[0.3, 0.4],
[0.5, 0.6],
[0.7, 0.8]])
print('The Original Matrix _____________\n',A)
# calculate pseudoinverse
B = pinv(A)
print('\n______________________________________pseudoinverse : \n',B)
# data reduction with svd

from scipy.linalg import svd
# define matrix
A = np.array([
[1,2,3,4,5,6,7,8,9,10],
[11,12,13,14,15,16,17,18,19,20],
[21,22,23,24,25,26,27,28,29,30]])
print('Real Matrix: .... \n',A)

# factorize
U, s, V = svd(A)

# create m x n Sigma matrix
Sigma = np.zeros((A.shape[0], A.shape[1]))

# populate Sigma with n x n diagonal matrix
Sigma[:A.shape[0], :A.shape[0]] = np.diag(s)

# select
n_elements = 2
Sigma = Sigma[:, :n_elements]
V = V[:n_elements, :]

# reconstruct
B = U.dot(Sigma.dot(V))
print('reconstructed Matrix :.......\n',B)

# transform
T = U.dot(Sigma)
print('______________transform one \n',T)
T = A.dot(V.T)
print('______________transform two \n',T)
# svd data reduction in scikit-learn

from sklearn.decomposition import TruncatedSVD
# define matrix
A = np.array([
[1,2,3,4,5,6,7,8,9,10],
[11,12,13,14,15,16,17,18,19,20],
[21,22,23,24,25,26,27,28,29,30]])
print('The real matrix: ....\n',A)
# create transform
svd = TruncatedSVD(n_components=2)
# fit transform
svd.fit(A)
# apply transform
result = svd.transform(A)
print('\ntransformed version of the matrix....... \n',result)
# vector mean

# define vector
v = np.array([1,2,3,4,5,6])
print('Vector ... \n',v)
# calculate mean
result = np.mean(v)
print('the mean of the values in the vector ..... \n',result)
# matrix means

# define matrix
M = np.array([
[1 ,2 ,3 ,4 ,5 ,6 ],
[1 ,2 ,3 ,4 ,5 ,6]])
print('_______________Full Matrix ...\n',M)
# column means
col_mean = np.mean(M, axis=0)
print('Columns mean values..... \n',col_mean)
# row means
row_mean = np.mean(M, axis=1)
print('Rows mean values....... \n',row_mean)
# vector variance

# define vector
v = np.array([1,2,3,4,5,6])
print('Vector ....... \n',v)
# calculate variance
result = np.var(v, ddof=1)
"""ddof ::::-
Delta Degrees of Freedom": 
                         the divisor used in the calculation is
                        ``N - ddof``, where ``N`` represents the number of elements. By default `ddof` is zero.
"""
print('Vector Variance ....... \n',result)
# matrix variances
# define matrix
M = np.array([
[1,2,3,4,5,6],
[1,2,3,4,5,6]])
print('Full Matrix..... \n',M)
# column variances
col_var = np.var(M, ddof=1, axis=0)
print('\nColumns variances.... \n',col_var)
# row variances
row_var = np.var(M, ddof=1, axis=1)
print('Rows variances ........ \n',row_var)

# column standard deviations
col_std = np.std(M, ddof=1, axis=0)
print('\ncolumn standard deviations ...\n',col_std)
# row standard deviations
row_std = np.std(M, ddof=1, axis=1)
print('row standard deviations .......\n',row_std)
# vector covariance

# define first vector
x = np.array([1,2,3,4,5,6,7,8,9])
print('First vector......... \n',x)
# define second covariance
y = np.array([9,8,7,6,5,4,3,2,1])
print('Second vector......... \n',y)

# calculate covariance
Sigma = np.cov(x,y)[0,1]
print('\ncovariance..... \n',Sigma)
# vector correlation

# define first vector
x = np.array([1,2,3,4,5,6,7,8,9])
print('first vector.............. \n',x)
# define second vector
y = np.array([9,8,7,6,5,4,3,2,1])
print('second vector............. \n',y)

# calculate correlation
corr = np.corrcoef(x,y)[0,1]
print('\ncorrelation Result ........... \n',corr)
# covariance matrix

# define matrix of observations
X = np.array([
[1, 5, 8],
[3, 5, 11],
[2, 4, 9],
[3, 6, 10],
[1, 5, 10]])
print('dataset with 5 observations across 3 features........... \n',X)
# calculate covariance matrix
Sigma = np.cov(X.T)
print('\n_____________________calculate covariance matrix .......\n',Sigma)
# principal component analysis

from numpy.linalg import eig
# define matrix
A = np.array([
[1, 2],
[3, 4],
[5, 6]])
print('Full Original Matrix .........\n',A)
# 1) >>>>>>>>>>>>>>> column means
M = np.mean(A.T, axis=1)
# 2) >>>>>>>>>>>>>>> center columns by subtracting column means
C = A - M
# 3) >>>>>>>>>>>>>>> calculate covariance matrix of centered matrix
V = np.cov(C.T)
# 4) >>>>>>>>>>>>>>> factorize covariance matrix
values, vectors = eig(V)
print('\nEigen Vectros........\n',vectors)
print('\nEigen Values........\n',values)

# project data
P = vectors.T.dot(C.T)
print('_________________________________Final Results-projection of the original matrix- ............\n',P.T)
# principal component analysis with scikit-learn
from sklearn.decomposition import PCA

# define matrix
A = np.array([
[1, 2],
[3, 4],
[5, 6]])
print('Full MAtrix .......\n',A)

# create the transform
pca = PCA(2) # n_components=3 must be between 0 and min(n_samples, n_features)=2 with svd_solver='full'

# fit transform
pca.fit(A)

# access values and vectors
print('\nprincipal components............... \n',pca.components_)
print('values .......................\n',pca.explained_variance_)

# transform data
B = pca.transform(A)
"""
Apply dimensionality reduction to A.

A is projected on the first principal components previously extracted
from a training set

"""
print('\n projection of the original matrix ......... \n',B)


# linear regression dataset
from numpy import array
from matplotlib import pyplot
# define dataset
data = array([
[0.05, 0.12],
[0.18, 0.22],
[0.31, 0.35],
[0.42, 0.38],
[0.5, 0.49]])
print('Real DataSet Matrix : \n',data)

# split into inputs and outputs
X, y = data[:,0], data[:,1]
X = X.reshape((len(X), 1))
print('\n_____________________\n Inpits:\n',X,'\n_________________________\n Outputs: \n',y,'\n_____________________')
# scatter plot
pyplot.scatter(X, y)
pyplot.show()
# direct solution to linear least squares
from numpy import array
from numpy.linalg import inv
from matplotlib import pyplot
# define dataset
data = array([
[0.05, 0.12],
[0.18, 0.22],
[0.31, 0.35],
[0.42, 0.38],
[0.5, 0.49]])
# split into inputs and outputs
X, y = data[:,0], data[:,1]
X = X.reshape((len(X), 1))
# linear least squares
b = inv(X.T.dot(X)).dot(X.T).dot(y)
print(b)
# predict using coefficients
yhat = X.dot(b)
# plot data and predictions
pyplot.scatter(X, y)
pyplot.plot(X, yhat, color='red')
pyplot.show()
# QR decomposition solution to linear least squares
from numpy import array
from numpy.linalg import inv
from numpy.linalg import qr
from matplotlib import pyplot
# define dataset
data = array([
[0.05, 0.12],
[0.18, 0.22],
[0.31, 0.35],
[0.42, 0.38],
[0.5, 0.49]])
# split into inputs and outputs
X, y = data[:,0], data[:,1]
X = X.reshape((len(X), 1))
# factorize
Q, R = qr(X)
b = inv(R).dot(Q.T).dot(y)
print(b)
# predict using coefficients
yhat = X.dot(b)
# plot data and predictions
pyplot.scatter(X, y)
pyplot.plot(X, yhat, color='red')
pyplot.show()
# SVD solution via pseudoinverse to linear least squares
from numpy import array
from numpy.linalg import pinv
from matplotlib import pyplot
# define dataset
data = array([
[0.05, 0.12],
[0.18, 0.22],
[0.31, 0.35],
[0.42, 0.38],
[0.5, 0.49]])
# split into inputs and outputs
X, y = data[:,0], data[:,1]
X = X.reshape((len(X), 1))

# calculate coefficients
b = pinv(X).dot(y)
print(b)

# predict using coefficients
yhat = X.dot(b)
# plot data and predictions
pyplot.scatter(X, y)
pyplot.plot(X, yhat, color='red')
pyplot.show()
# least squares via convenience function
from numpy import array
from numpy.linalg import lstsq
from matplotlib import pyplot
# define dataset
data = array([
[0.05, 0.12],
[0.18, 0.22],
[0.31, 0.35],
[0.42, 0.38],
[0.5, 0.49]])
# split into inputs and outputs
X, y = data[:,0], data[:,1]
X = X.reshape((len(X), 1))
# calculate coefficients
b, residuals, rank, s = lstsq(X, y)
print(b)
# predict using coefficients
yhat = X.dot(b)
# plot data and predictions
pyplot.scatter(X, y)
pyplot.plot(X, yhat, color='yellow')
pyplot.show()