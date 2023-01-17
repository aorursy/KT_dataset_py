

#Attributes of arrays: Determining the size, shape, memory consumption, and data types of arrays

#Indexing of arrays: Getting and setting the value of individual array elements

#Slicing of arrays: Getting and setting smaller subarrays within a larger array

#Reshaping of arrays: Changing the shape of a given array

#Joining and splitting of arrays: Combining multiple arrays into one, and splitting one array into many
import numpy

numpy.__version__
import numpy as np
a=np.random.randint(10,size=6)
a
b=np.random.randint(6,size=10)
b
c=np.random.randint(10,size=(3,2))
c
d=np.random.randint(10,size=(3,3))
d
e=np.random.randint(10,size=(3,3,3))
e
f=np.random.randint(10,size=(3,3,1))
f
print("dtype:", f.dtype)
print("itemsize:", a.itemsize, "bytes")

print("nbytes:", b.nbytes, "bytes")

print("itemsize:", c.itemsize, "bytes")

print("nbytes:", d.nbytes, "bytes")

print("itemsize:", e.itemsize, "bytes")

print("nbytes:", f.nbytes, "bytes")

a
a[0]
a[-1]
a[-2]
a[-3]
b
c
c[0,0]
d
d[1,1]
d[0,0]
d[0,1]
d[0,2]
d[1,0]
d[1,1]
d[1,2]
d[2,0]
d[2,1]
d[2,2]
a
a[:5]  # first five elements
b
b[:3]#first three
b[3:]#after three element
b[3:7]#middle element
b[::3]
b[:1:2]
b[::-1]
d
d[:2,:3]#first two rows
d[:3, ::1]  # all rows, every other column
grid = np.arange(1, 10)

grid
grid = np.arange(1, 10).reshape((3, 3))

print(grid)
x = np.array([1, 2, 3])

print(x)



# row vector via reshape

x.reshape((1, 3))
# column vector via reshape

x.reshape((3, 1))
x = np.array([1, 2, 3])

y = np.array([3, 2, 1])

np.concatenate([x, y])
z = [99, 99, 99]

print(np.concatenate([x, y, z]))
grid = np.array([[1, 2, 3],

                 [4, 5, 6]])
grid
# concatenate along the first axis

np.concatenate([grid, grid])
x = [1, 2, 3, 99, 99, 3, 2, 1]

x1, x2, x3 = np.split(x, [3, 5])

print(x1, x2, x3)
j=[1,2,3,4,5,6,7,8,9,10,11,12,13]
j
j1,j2,j3=np.split(x,[2,4])
print(j1,j2,j3)
upper, lower = np.vsplit(grid, [3])

print(upper)

print(lower)
# Create a 1D ndarray that contains only integers

x = np.array([1, 2, 3, 4, 5])

print('x = ', x) # x = [1 2 3 4 5]

print('x has dimensions:', x.shape) # x has dimensions: (5,)

print('The elements in x are of type:', x.dtype) # The elements in x are of type: int64



# Create a rank 2 ndarray that only contains integers

Y = np.array([[1,2,3],[4,5,6],[7,8,9], [10,11,12]])

print('Y has dimensions:', Y.shape) # Y has dimensions: (4, 3)

print('Y has a total of', Y.size, 'elements') # Y has a total of 12 elements

print('Y is an object of type:', type(Y)) # Y is an object of type: class 'numpy.ndarray'

print('The elements in Y are of type:', Y.dtype) # The elements in Y are of type: int64
x
x.shape
x.dtype
Y
Y.shape
Y.size
type(Y)
Y.dtype
# Specify the dtype when creating the ndarray

x = np.array([1.5, 2.2, 3.7, 4.0, 5.9], dtype = np.int64)
x
# Save the array into a file

np.save('my_array', x)



# Load the saved array from current directory

y = np.load('my_array.npy')

# Create ndarray using built-in functions

# 3 x 4 ndarray full of zeros

# np.zeros(shape)

X = np.zeros((3,4))
X
# a 3 x 2 ndarray full of ones

# np.ones(shape)

X = np.ones((3,2))
X
# 2 x 3 ndarray full of fives

# np.full(shape, constant value)

X = np.full((2,3), 5)
X
# Identity Matrix

# Since all Identity Matrices are square, the np.eye() function only takes a single integer as an argument

# 5 x 5 Identity matrix

X = np.eye(5)
X
# Diagonal Matrix

# 4 x 4 diagonal matrix that contains the numbers 10,20,30, and 50 on its main diagonal

X = np.diag([10,20,30,50])
X
# Arange

# rank 1 ndarray that has sequential integers from 0 to 9

# x = [0 1 2 3 4 5 6 7 8 9]

x = np.arange(10)



# rank 1 ndarray that has sequential integers from 4 to 9

# [start, stop)

# x = [4 5 6 7 8 9]

x = np.arange(4,10)



# rank 1 ndarray that has evenly spaced integers from 1 to 13 in steps of 3.

# np.arange(start,stop,step)

# x = [ 1 4 7 10 13]

x = np.arange(1,14,3)
x
x = np.arange(4,10)
x
x = np.arange(1,14,3)
x
# Linspace

# Even though the np.arange() function allows for non-integer steps,

# such as 0.3, the output is usually inconsistent, due to the finite

# floating point precision. For this reason, in the cases where

# non-integer steps are required, it is usually better to use linspace()

# becayse np.linspace() uses the number of elements we want in a

# particular interval, instead of the step between values.

# linspace returns N evenly spaced numbers over the closed interval [start, stop]

# np.linspace(start, stop, N)

# x = [ 0. 2.77777778 5.55555556 8.33333333 11.11111111 13.88888889 16.66666667 19.44444444 22.22222222 25. ]

x = np.linspace(0,25,10)

x = np.linspace(0,25,10)
x
# Reshape

# np.reshape(ndarray, new_shape)

# converts the given ndarray into the specified new_shape

x = np.arange(20)

x = np.reshape(x, (4,5))

# or

x = np.arange(20).reshape(4, 5) # does the same thing as above

# and the same thing with with linshape

y = np.linspace(0,50,10, endpoint=False).reshape(5,2)

# One great feature about NumPy, is that some functions can also be

# applied as methods. This allows us to apply different functions in

# sequence in just one line of code
z=np.arange(10)
z
x = np.arange(20).reshape(4, 5)
x
# Slicing

# ndarray[start:end]

# ndarray[start:]

# ndarray[:end]

# ndarray[<start>:<stop>:<step>]



# In methods one and three, the end index is excluded [,)

X = np.arange(20).reshape(4, 5)



# select all the elements that are in the 2nd through 4th rows and in the 3rd to 5th columns

Z = X[1:4,2:5]

# or

Z = X[1:,2:5]



# elements = a_list[<start>:<stop>:<step>]

# select all the elements in the 3rd row

v = X[2,:] # v = [10 11 12 13 14]

# select all the elements in the 3rd column

q = X[:,2] # q = [ 2 7 12 17]

# select all the elements in the 3rd column but return a rank 2 ndarray

R = X[:,2:3]

'''

[[ 2]

 [ 7]

 [12]

 [17]]

'''
# Note: Slicing creates a view, not a copy

# when we make assignments, such as: Z = X[1:4,2:5]

# the slice of the original array X is not copied in the variable Z.

# Rather, X and Z are now just two different names for the same ndarray.

# We say that slicing only creates a view of the original array.

# This means if we make changes to Z, X changes as well.
# Random

# 3 x 3 ndarray with random floats in the half-open interval [0.0, 1.0).

# np.random.random(shape)

X = np.random.random((3,3))

# np.random.randint(start, stop, size = shape)

# [start, stop)

X = np.random.randint(4,15,size=(3,2))



# create ndarrays with random numbers that satisfy certain statistical properties

# 1000 x 1000 ndarray of random floats drawn from normal (Gaussian)

# distribution with a mean of zero and a standard deviation of 0.1.

# np.random.normal(mean, standard deviation, size=shape)

X = np.random.normal(0, 0.1, size=(1000,1000))
# Mutability

# Change ndarray

x[3] = 20

X[0,0] = 20
X
# Delete

# np.delete(ndarray, elements, axis)

x = np.array([1, 2, 3, 4, 5])

# delete the first and fifth element of x

x = np.delete(x, [0,4])



Y = np.array([[1,2,3],[4,5,6],[7,8,9]])

# delete the first row of Y

w = np.delete(Y, 0, axis=0)

# delete the first and last column of Y

v = np.delete(Y, [0,2], axis=1)
x
Y
w
v
# Append

# np.append(ndarray, elements, axis)

# append the integer 6 to x

x = np.append(x, 6)

# append the integer 7 and 8 to x

x = np.append(x, [7,8])

# append a new row containing 7,8,9 to y

v = np.append(Y, [[10,11,12]], axis=0)

# append a new column containing 9 and 10 to y

q = np.append(Y,[[13],[14],[15]], axis=1)
x
v
q
# Insert

# np.insert(ndarray, index, elements, axis)

# inserts the given list of elements to ndarray right before

# the given index along the specified axis

x = np.array([1, 2, 5, 6, 7])

Y = np.array([[1,2,3],[7,8,9]])

# insert the integer 3 and 4 between 2 and 5 in x. 

x = np.insert(x,2,[3,4])

# insert a row between the first and last row of Y

w = np.insert(Y,1,[4,5,6],axis=0)

# insert a column full of 5s between the first and second column of Y

v = np.insert(Y,1,5, axis=1)
x
Y
x
w
v
# Stacking

# NumPy also allows us to stack ndarrays on top of each other,

# or to stack them side by side. The stacking is done using either

# the np.vstack() function for vertical stacking, or the np.hstack()

# function for horizontal stacking. It is important to note that in

# order to stack ndarrays, the shape of the ndarrays must match.

x = np.array([1,2])

Y = np.array([[3,4],[5,6]])

z = np.vstack((x,Y)) # [[1,2], [3,4], [5,6]]

w = np.hstack((Y,x.reshape(2,1))) # [[3,4,1], [5,6,2]]
# Copy

# if we want to create a new ndarray that contains a copy of the

# values in the slice we need to use the np.copy()

# create a copy of the slice using the np.copy() function

Z = np.copy(X[1:4,2:5])

#  create a copy of the slice using the copy as a method

W = X[1:4,2:5].copy()
Z
W
# Extract elements along the diagonal

d0 = np.diag(X)

# As default is k=0, which refers to the main diagonal.

# Values of k > 0 are used to select elements in diagonals above

# the main diagonal, and values of k < 0 are used to select elements

# in diagonals below the main diagonal.

d1 = np.diag(X, k=1)

d2 = np.diag(X, k=-1)
d1
d2
# Find Unique Elements in ndarray

u = np.unique(X)
u
# Boolean Indexing

X = np.arange(25).reshape(5, 5)

print('The elements in X that are greater than 10:', X[X > 10])

print('The elements in X that less than or equal to 7:', X[X <= 7])

print('The elements in X that are between 10 and 17:', X[(X > 10) & (X < 17)])



# use Boolean indexing to assign the elements that

# are between 10 and 17 the value of -1

X[(X > 10) & (X < 17)] = -1
# Set Operations

x = np.array([1,2,3,4,5])

y = np.array([6,7,2,8,4])

print('The elements that are both in x and y:', np.intersect1d(x,y))

print('The elements that are in x that are not in y:', np.setdiff1d(x,y))

print('All the elements of x and y:',np.union1d(x,y))
x
y
# Sorting

# When used as a function, it doesn't change the original ndarray

s = np.sort(x)

# When used as a method, the original array will be sorted

x.sort()



# sort x but only keep the unique elements in x

s = np.sort(np.unique(x))



# sort the columns of X

s = np.sort(X, axis = 0)



# sort the rows of X

s = np.sort(X, axis = 1)
s
s
# NumPy allows element-wise operations on ndarrays as well as

# matrix operations. In order to do element-wise operations,

# NumPy sometimes uses something called Broadcasting.

# Broadcasting is the term used to describe how NumPy handles

# element-wise arithmetic operations with ndarrays of different shapes.

# For example, broadcasting is used implicitly when doing arithmetic

# operations between scalars and ndarrays.

x = np.array([1,2,3,4])

y = np.array([5.5,6.5,7.5,8.5])

np.add(x,y)

np.subtract(x,y)

np.multiply(x,y)

np.divide(x,y)



# in order to do these operations the shapes of the ndarrays

# being operated on, must have the same shape or be broadcastable

X = np.array([1,2,3,4]).reshape(2,2)

Y = np.array([5.5,6.5,7.5,8.5]).reshape(2,2)

np.add(X,Y)

np.subtract(X,Y)

np.multiply(X,Y)

np.divide(X,Y)



# apply mathematical functions to all elements of an ndarray at once

np.exp(x)

np.sqrt(x)

np.power(x,2)
x
y
np.add(x,y)
# Statistical Functions

print('Average of all elements in X:', X.mean())

print('Average of all elements in the columns of X:', X.mean(axis=0))

print('Average of all elements in the rows of X:', X.mean(axis=1))

print()

print('Sum of all elements in X:', X.sum())

print('Standard Deviation of all elements in X:', X.std())

print('Median of all elements in X:', np.median(X))

print('Maximum value of all elements in X:', X.max())

print('Minimum value of all elements in X:', X.min())
# Broadcasting

# NumPy is working behind the scenes to broadcast 3 along the ndarray

# so that they have the same shape. This allows us to add 3 to each

# element of X with just one line of code.

print(4*X)

print(4+X)

print(4-X)

print(4/X)

# NumPy is able to add 1 x 3 and 3 x 1 ndarrays to 3 x 3 ndarrays

# by broadcasting the smaller ndarrays along the big ndarray so that

# they have compatible shapes. In general, NumPy can do this provided

# that the smaller ndarray can be expanded to the shape of the larger

# ndarray in such a way that the resulting broadcast is unambiguous.

x = np.array([1,2,3])

Y = np.array([[1,2,3],[4,5,6],[7,8,9]])

Z = np.array([1,2,3]).reshape(3,1)

print(x + Y)

print(Z + Y)
