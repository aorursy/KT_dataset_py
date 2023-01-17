import numpy as np

np.dot(3,4)

# Create a rank 1 array

a = np.array([0, 1, 2])

print(type(a))

# this will print the dimension of the array

print(a.shape)

print(a[0])

print(a[1])

print(a[2])

# Change an element of the array

a[0] = 5

print(a)

# Create a rank 2 array

b = np.array([[0,1,2],[3,4,5]])

print(b.shape)

print(b)

print(b[0, 0], b[0, 1], b[1, 0])
# Create a 3x3 array of all zeros

a = np.zeros((3,3))

print(a)

# Create a 2x2 array of all ones

b = np.ones((2,2))

print(b)

# Create a 3x3 constant array



c = np.full((3,3), 7)

print(c)

# Create a 3x3 array filled with random values

d = np.random.random((3,3))

print(d)

# Create a 3x3 identity matrix

e = np.eye(3)

print(e)

# convert list to array

f = np.array([2, 3, 1, 0])

print(f)

# arange() will create arrays with regularly incrementing values

g = np.arange(20)

print(g)

# note mix of tuple and lists

h = np.array([[0, 1,2.0],[0,0,0],(1+1j,3.,2.)])

print(h)

# create an array of range with float data type

i = np.arange(1, 8, dtype=np.float)

print(i)

# linspace() will create arrays with a specified number of items which are

# spaced equally between the specified beginning and end values j= np.linspace(start, stop, num)

j = np.linspace(2, 4, 5)

print(j)
# Let numpy choose the datatype

x = np.array([0, 1])

y = np.array([2.0, 3.0])

# Force a particular datatype

z = np.array([5, 6], dtype=np.int64)

print (x.dtype, y.dtype, z.dtype)
# An exemplar array

arr = np.array([[-1, 2, 0, 4],

[4, -0.5, 6, 0],

[2.6, 0, 7, 8],

[3, -7, 4, 2.0]])

# Slicing array

temp = arr[:2, ::2]

print ("Array with first 2 rows and alternate columns(0 and 2):\n", temp)

# Integer array indexing example

temp = arr[[0, 1, 2, 3], [3, 2, 1, 0]]

print ("\nElements at indices (0, 3), (1, 2), (2, 1),(3, 0):\n", temp)

# boolean array indexing example

cond = arr > 0 # cond is a boolean array

temp = arr[cond]

print ("\nElements greater than 0:\n", temp)
a = np.array([1, 2, 5, 3])

# add 1 to every element

print ("Adding 1 to every element:", a+1)

# subtract 3 from each element

print ("Subtracting 3 from each element:", a-3)

# multiply each element by 10

print ("Multiplying each element by 10:", a*10)

# square each element

print ("Squaring each element:", a**2)

# modify existing array

a *= 2

print ("Doubled each element of original array:", a)

# transpose of array

a = np.array([[1, 2, 3], [3, 4, 5], [9, 6, 0]])

print ("\nOriginal array:\n", a)

print ("Transpose of array:\n", a.T)
arr = np.array([[1, 5, 6],

[4, 7, 2],

[3, 1, 9]])

# maximum element of array

print ("Largest element is:", arr.max())

print ("Row-wise maximum elements:",arr.max(axis = 1))

# minimum element of array

print ("Column-wise minimum elements:",arr.min(axis = 0))

# sum of array elements

print ("Sum of all array elements:",arr.sum())

# cumulative sum along each row

print ("Cumulative sum along each row:\n",arr.cumsum(axis = 1))
# Python program to demonstrate

# binary operators in Numpy

import numpy as np

a = np.array([[1, 2],

[3, 4]])

b = np.array([[4, 3],

[2, 1]])

# add arrays

print ("Array sum:\n", a + b)

# multiply arrays (elementwise multiplication)

print ("Array multiplication:\n", a*b)

# matrix multiplication

print ("Matrix multiplication:\n", a.dot(b))
# create an array of sine values

a = np.array([0, np.pi/2, np.pi])

print ("Sine values of array elements:", np.sin(a))

# exponential values

a = np.array([0, 1, 2, 3])

print ("Exponent of array elements:", np.exp(a))

# square root of array values

print ("Square root of array elements:", np.sqrt(a))
arr = np.array([[-1, 2, 0, 4],

[4, -0.5, 6, 0],

[2.6, 0, 7, 8],

[3, -7, 4, 2.0]])

inverse = np.linalg.inv(arr)

inverse
#Dot product of two arrays

a = np.array([[1, 2],[3, 4]]) #matrix

v = np.array([4, 3]) #vector

print('dot product: \n',np.dot(a, v))
''''Compute the dot product of two or more arrays in a single function

call, while automatically selecting the fastest evaluation order.'''

b = np.array([[4, 3],[2, 1]])

c = np.array([[5, 6],[7, 8]])

print(np.linalg.multi_dot([a,b,c]))
#Return the dot product of two vectors.

v2=[1,2]

print(np.vdot(v, v2))
#Inner product of two arrays.

print(np.inner(v,v2))
#Compute the outer product of two vectors.

print(np.outer(v,v2))
#Matrix product of two arrays.

print(np.matmul(a, b))
#Compute tensor dot product along specified axes for arrays >= 1-D

print(np.tensordot(a, b))
#Evaluates the Einstein summation convention on the operands

print(np.einsum('ij,jh',a,b))
'''Evaluates the lowest cost contraction order for an einsum

expression by considering the creation of intermediate arrays.'''

print(np.einsum_path('ij,jh',a,b))
#Raise a square matrix to the (integer) power n.

print(np.linalg.matrix_power(a, 10))
#Matrix Eigen Values

#Compute the eigenvalues and right eigenvectors of a square array.

print(np.linalg.eig(a))
'''Return the eigenvalues and eigenvectors of a complex Hermitian (conjugate symmetric) or a

real symmetric matrix.'''

print(np.linalg.eigh(a))
#Compute the eigenvalues of a general matrix.

print(np.linalg.eigvals(a))