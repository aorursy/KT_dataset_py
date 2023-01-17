### Import Numpy

# To install on your local machine, open a terminal and type:

# Windows/Mac:

# python -m pip install numpy

# Linux:

# python3 -m pip install numpy

import numpy as np
### Creating an ndarray

## Can pass any iterable into np.array() to transform it to an ndarray

## Numpy arrays are statically typed, unlike python lists!

x = np.array(['a', 'b', 'c'])

print(x)

# array(['a', 'b', 'c'], dtype='<U1')



## Numpy version of range, generates an ndarray instead

x = np.arange(start=0, stop=3, step=1)

print(x)

# array([0, 1, 2])



## np.arange allows for non integer step sizes

x = np.arange(start=0, stop=1, step=.1)

print(x)

# array([0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])



## Many functions to create quick arrays exist

x = np.zeros(shape=6)

print(x)

# array([0. 0. 0. 0. 0. 0.])



## Generating n dimensional is usually done by passing a tuple

#  into an array generation function.

x = np.zeros(shape=(3, 5))

print(x)

# array([[0., 0., 0., 0., 0.],

#        [0., 0., 0., 0., 0.],

#        [0., 0., 0., 0., 0.]])



## Numpy can generate large amounts of random numbers very quickly

x = np.random.uniform(0, 1, size=10)

print(x)

# array([0.48908712, 0.21474925, 0.71589095, 0.74606879, 0.0992887 ,

#        0.33601409, 0.41361811, 0.85178078, 0.21592613, 0.8155579 ])
### Reshaping ndarrays

x = np.arange(1, 13)

print(x)

# array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12])



## Reshape by passing tuple w/ (n_rows, n_cols)

y = x.reshape((3, 4))

print(y)

# array([[ 1,  2,  3,  4],

#        [ 5,  6,  7,  8],

#        [ 9, 10, 11, 12]])



## Transpose matrix

x = x.T

print(x)

# array([[ 1,  5,  9],

#        [ 2,  6, 10],

#        [ 3,  7, 11],

#        [ 4,  8, 12]])



## When reshaping can use -1 as parameter, the -1 is 

# switched to the maximum value it can be to generate 

# a valid array

x = np.random.power(7, size=4)

print(x)

# array([0.73801263, 0.8526897 , 0.88994279, 0.93148266])

print(x.T)

# array([0.73801263, 0.8526897 , 0.88994279, 0.93148266])

print(x.reshape((-1, 1)))

# array([[0.73801263],

#        [0.8526897 ],

#        [0.88994279],

#        [0.93148266]])
### Combining ndarrays

a = np.arange(6)

b = a.reshape((2, -1))

print(a)

print(b)

# array([0, 1, 2, 3, 4, 5])

# array([[0, 1, 2],

#        [3, 4, 5]])



## Concatenate essentially glues arrays together along axis

x = np.concatenate((b, b), axis=0)

print(x)

# array([[0, 1, 2],

#        [3, 4, 5],

#        [0, 1, 2],

#        [3, 4, 5]])



x = np.concatenate((b, b), axis=1)

print(x)

# array([[0, 1, 2, 0, 1, 2],

#        [3, 4, 5, 3, 4, 5]])



## vstack is an alias to concatenate: axis=0

x = np.vstack((b, b))

print(x)

# array([[0, 1, 2],

#        [3, 4, 5],

#        [0, 1, 2],

#        [3, 4, 5]])



## hstack is an alias to concatenate: axis=1

x = np.hstack((b, b))

print(x)

# array([[0, 1, 2, 0, 1, 2],

#        [3, 4, 5, 3, 4, 5]])
### Indexing & Slicing ndarrays

# [row, column, channel]

x = np.arange(12).reshape((3, 4))

print(x)

# array([[ 0,  1,  2,  3],

#        [ 4,  5,  6,  7],

#        [ 8,  9, 10, 11]])

print(x[1, 2])

# 6



x = np.arange(12).reshape((2, 2, 3))

print(x)

# array([[[ 0,  1,  2],

#         [ 3,  4,  5]],

#

#        [[ 6,  7,  8],

#         [ 9, 10, 11]]])

print(x[0, 1, 2])

# 5



## Multiple indexing, passing a list instead of an int

# Note: values are returned as ndarray

x = np.arange(0, 18, 3)

print(x)

# array([ 0,  3,  6,  9, 12, 15])

print(x[[1, 3, 5]])

# array([ 3,  9, 15])



## np.where generates indexing matrix based on what values are true

#  in given matrix

x = np.arange(10).reshape((2, -1))

print(x)

# array([[0, 1, 2, 3, 4],

#        [5, 6, 7, 8, 9]])

is_odd = x % 2

# array([[0, 1, 0, 1, 0],

#        [1, 0, 1, 0, 1]], dtype=int32)

locs = np.where(is_odd)

print(locs)

# (array([0, 0, 1, 1, 1], dtype=int64), array([1, 3, 0, 2, 4], dtype=int64))

print(x[locs])

# array([1, 3, 5, 7, 9])



## Slicing, the same as pure python with multiple dimensions.

x = np.arange(5)

print(x)

# array([0, 1, 2, 3, 4])

print(x[::1])

# array([0, 1, 2, 3, 4])

print(x[::2])

# array([0, 2, 4])

print(x[::-1])

# array([4, 3, 2, 1, 0])



x = np.arange(5) + np.arange(5).reshape((-1, 1))

print(x)

# array([[0, 1, 2, 3, 4],

#        [1, 2, 3, 4, 5],

#        [2, 3, 4, 5, 6],

#        [3, 4, 5, 6, 7],

#        [4, 5, 6, 7, 8]])

print(x[1:4])

# array([[1, 2, 3, 4, 5],

#        [2, 3, 4, 5, 6],

#        [3, 4, 5, 6, 7]])

print(x[:, 1:4])

# array([[1, 2, 3],

#        [2, 3, 4],

#        [3, 4, 5],

#        [4, 5, 6],

#        [5, 6, 7]])

print(x[1:4, 1:4])

# array([[2, 3, 4],

#        [3, 4, 5],

#        [4, 5, 6]])
### Broadcasting

## Broadcasting is numpy's way of performing operations between

#  arrays of different lengths



## Quickly multiplying array by scalar

x = np.ones(shape=3) * 6

print(x)

# array([6., 6., 6.])



## Adding vector a with shape (1, 3) and b with shape (3, 1) = 

#  x with shape (3, 3)

a = np.arange(3)

b = np.arange(3).reshape((-1, 1))

print(a)

# array([0, 1, 2])

print(b)

# array([[0],

#        [1],

#        [2]])

x = a + b

print(x)

# array([[0, 1, 2],

#        [1, 2, 3],

#        [2, 3, 4]])



## Longer demonstration

x = np.array([-3, 0, 3]) + np.zeros(shape=(3, 1))

print(x)

# array([[-3.,  0.,  3.],

#        [-3.,  0.,  3.],

#        [-3.,  0.,  3.]])

y = np.array([-10, 0, 10])

print(y)

# array([-10,   0,  10])

print(x * y)

# array([[30.,  0., 30.],

#        [30.,  0., 30.],

#        [30.,  0., 30.]])

y = y.reshape((-1, 1))

print(y)

# array([[-10],

#        [  0],

#        [ 10]])

print(x * y)

# array([[ 30.,  -0., -30.],

#        [ -0.,   0.,   0.],

#        [-30.,   0.,  30.]])



## Arithmetic, Comparisons and set = are broadcastable operations

# generating a crosshatch pattern

x = np.arange(5) + np.zeros(shape=(3, 1))

print(x)

# array([[0., 1., 2., 3., 4.],

#        [0., 1., 2., 3., 4.],

#        [0., 1., 2., 3., 4.]])

y = np.array([[1], [2], [3]])

print(y)

# array([[1],

#        [2],

#        [3]])



z = x + y

print(z)

# array([[1., 2., 3., 4., 5.],

#        [2., 3., 4., 5., 6.],

#        [3., 4., 5., 6., 7.]])



# ~ inverts boolean values

idx = ~np.bool_(z % 2)

print(idx)

# array([[False,  True, False,  True, False],

#        [ True, False,  True, False,  True],

#        [False,  True, False,  True, False]])



z[idx] = 0

print(z)

# array([[1., 0., 3., 0., 5.],

#        [0., 3., 0., 5., 0.],

#        [3., 0., 5., 0., 7.]])



crosshatch = z / z

print(crosshatch)

# array([[ 1., nan,  1., nan,  1.],

#        [nan,  1., nan,  1., nan],

#        [ 1., nan,  1., nan,  1.]])



crosshatch[np.isnan(crosshatch)] = 0

print(crosshatch)

# array([[1., 0., 1., 0., 1.],

#        [0., 1., 0., 1., 0.],

#        [1., 0., 1., 0., 1.]])
### Memory Nuances

## Slicing and indexing values of ndarray return actual

## memory locations that can be used to modify original

a = np.arange(10).reshape((2, 5))

print(a)

# array([[0, 1, 2, 3, 4],

#        [5, 6, 7, 8, 9]])

b = a[::-1, ::-1]

print(b)

# array([[9, 8, 7, 6, 5],

#        [4, 3, 2, 1, 0]])

a[0] += 1

print(a)

# array([[1, 2, 3, 4, 5],

#        [5, 6, 7, 8, 9]])

print(b)

# array([[9, 8, 7, 6, 5],

#        [5, 4, 3, 2, 1]])



## np.copy is a quick way to remove this issue

a = np.arange(10).reshape((2, 5))

print(a)

# array([[0, 1, 2, 3, 4],

#        [5, 6, 7, 8, 9]])

b = np.copy(a)[::-1, ::-1]

print(b)

# array([[9, 8, 7, 6, 5],

#        [4, 3, 2, 1, 0]])

a[0] += 1

print(a)

# array([[1, 2, 3, 4, 5],

#        [5, 6, 7, 8, 9]])

print(b)

# array([[9, 8, 7, 6, 5],

#        [4, 3, 2, 1, 0]])
### Numpy Statistics

x = np.random.normal(loc=12, size=10000)

print(np.mean(x))

# 12.004210965789426



x = np.random.randint(0, 100, size=100000)

print(np.median(x))

# 49.0
### Masked Arrays

# data[any], mask[bool], fill_value[any] <- value given w/ array-array comparison 

# when location is masked(?)

x = np.ma.array(np.arange(10), mask=[False] * 10)

print(x)

# masked_array(data=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],

#              mask=[False, False, False, False, False, False, False, False,

#                    False, False],

#        fill_value=999999)



x.mask[np.where(x % 2)] = True

print(x)

# masked_array(data=[0, --, 2, --, 4, --, 6, --, 8, --],

#              mask=[False,  True, False,  True, False,  True, False,  True,

#                    False,  True],

#        fill_value=999999)



print(x >= 0)

# masked_array(data=[True, --, True, --, True, --, True, --, True, --],

#              mask=[False,  True, False,  True, False,  True, False,  True,

#                    False,  True],

#        fill_value=999999)



print(np.sum(x))

# 20
### General Functions

a = list(range(5))

print(a)

# [0, 1, 2, 3, 4]

b = a[::-1]

print(b)

# [4, 3, 2, 1, 0]

for i, value in enumerate(b):

    a[i] += value



print(a)

# [4, 4, 4, 4, 4]



a = np.arange(10).reshape((2, 5))

print(a)

# array([[0, 1, 2, 3, 4],

#        [5, 6, 7, 8, 9]])

b = np.copy(a)[::-1, ::-1]

print(b)

# array([[9, 8, 7, 6, 5],

#        [4, 3, 2, 1, 0]])

for idx, value in np.ndenumerate(b):  # n dimensional enumerate, idx=tuple

    a[idx] += value

print(a)

# array([[9, 9, 9, 9, 9],

#        [9, 9, 9, 9, 9]])



x = np.arange(12).reshape((2, 2, 3))

print(x)

# array([[[ 0,  1,  2],

#         [ 3,  4,  5]],



#        [[ 6,  7,  8],

#         [ 9, 10, 11]]])

print(np.ravel(x))

# array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])



x = np.arange(7)

print(x)

# array([0, 1, 2, 3, 4, 5, 6])

y = np.minimum(x, x[::-1])

print(y)

# array([0, 1, 2, 3, 2, 1, 0])

print(np.argmax(y))  # -> index of max value

# 3