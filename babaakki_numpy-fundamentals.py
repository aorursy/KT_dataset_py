#importing Numpy module as np

#np is a short name that will be used throughout, whenever we refer numpy

import numpy as np
x = np.float(25)

x, type(x)
y = np.int32(4)

y, type(y)
z = np.complex(1,2)

z, type(z)
#Creating array from list

x = np.array([2,3,1,0])

x, type(x), x.dtype
#Creating array from list with type of elements specified

x = np.array([2,3,1,0], dtype=np.float16)

x, type(x), x.dtype
#Creating array from note mix of tuple and lists, and types

x = np.array([[1,2.0],[0,0],(1+1j,3.)])

x, type(x), x.dtype
#Creating array of more than one dimension

x = np.array([[1,2,3], [4,5,6]])

x, x.ndim
x = np.array([1,2,3])

y = np.array([1,2,3], ndmin=2)

x.ndim, y.ndim
#np.zeroes :: Creates array of zeroes of desired dimension

#default dtype is float64

z = np.zeros((5, 2))

z
#np.ones :: Creates array of ones of desired dimension

o = np.ones((2, 5), dtype=np.int32)

o
#np.arrange() :: Creates array of regularly incrementing values

#start by-default 0

np.arange(10)
#start is 2 inclusive and end is 10 exclusive

np.arange(2, 10)
#start 2, end 5, interval 0.5

np.arange(2, 5, 0.5)
#np.linspace() :: Creates an array of desired number of elements within a range

np.linspace(0, 100, 11, dtype=np.int32)
#Creates array of given shape with random values 

#from uniform distribution over [0, 1)



np.random.rand(2, 5)
#Array of random integers of size (5, 2)

#Values between [0, 100)

np.random.randint(0, 100, (5,2))
#Creates array of random floats in the half-open interval [0.0, 1.0).

np.random.sample((2,5))
Z = np.random.rand(5,4)
#Dimension

Z.ndim
#shape

Z.shape
#size

Z.size
#dtype of elements

Z.dtype, Z.dtype.name
Z
type(Z)
#Printing ndarray :: Python print function

print(Z)
x = np.array([10, 20, 30, 40])

y = np.linspace(0, 3, 4, dtype=np.int32)

x, y
#Element-wise addition

a = x + y

a
#Element-wise multiplication

m = x * y

m
#Using in-built mathematical function (sin) and constant (pi)

np.sin(np.pi*y)
s = y**2

s
r = np.random.randint(0, 100, 11)

r
#Some statistical Measures

np.sum(r), np.max(r), np. min(r), np.mean(r), np.median(r)

#Explore rest
np.exp(np.random.randn(2,4))
#Results boolean array of those satisfying the condition

r>50
n = np.arange(12)

n, n.shape
N = n.reshape(3,4)

N, N.shape
#2-d array or Matrix

np.zeros((4,4))
np.ones((3,3,3), dtype=np.int32)
#Diagonal Matrix

np.diag(np.arange(5))
#Similar to mesh grid in MATLAB

x, y = np.mgrid[0:5, 0:5]
x
y
print(n)
# n is 1d array, [index]

#index ::  0 to len(array)-1

#negitive indices starts from end

n[5], n[-1]
#Manipulating Value at a particular position

n[0] = -1

print(n)
print(N)
#N is 2d array

#[row, col]

N[1,1]
#If only one value inside the [], returns the entire row

N[2]

#Equivalent to N[2:]
N[2:]  #Row 3
#Fetching an entire column

N[:,1]  #Col 2
#Assigning 23 to element at location: 2nd row and 4th column

N[2,3] = 23

N
#Accessing elements from index 2 to index 4

n[2:5]
n[3:5] = [34, 45]

n
#No lower bound and upper bound involved, stride is 2

n[::2]
#Index Slicing Works Similarly for multi-dimensional Array

N[:,:]
#Extracting a 3x3 Matrix from N

M = N[0:3, 0:3]

print(M)

print(M.ndim)

print(M.shape)

print(M.size)
M[0:2, 0:2] = np.arange(1,5).reshape(2,2)

M
Mat = np.arange(20).reshape(5,4)

print(Mat)
print(Mat[::2, ::2])
a = np.array([1, 2, 3, 4])

b = 5

print(a*b)
x = np.arange(4)

xx = x.reshape(4,1)

y = np.ones(5)

z = xx + y

print(z)
xx.shape, y.shape, z.shape
#Vectors :: 1d Array

v1 = np.array([2, 5, 0, 4])

v2 = np.array([1, 9, 9, 1])
#Vector Sum

v1 + v2
#Multiplication with Scalar

v1 * 5
#Dot Product

v1.dot(v2)
M = np.matrix(N[0:3,0:3])

M
#Matrix Multiplication 1

M2 = np.dot(M, M)

M2
#Matrix Multiplication 2

M*M
V = np.matrix(v1[0:3]).reshape(3,1)

V
#Matrix Vector Multiplication

print(M*V)
#Transpose

M.T
#Transpose 2

np.transpose(M)
#Inverse of a Matrix

np.linalg.inv(M)
#Identity Matrix

np.eye(5)