# User defined function to get square of number
def fn_sq(x):
    return(x ** 2)
# Above equivalent funtion via Lambda
lambda x: x ** 2
# Lambda funtion for summation of two numbers
lambda x, y: x + y
# Map funtion applies a lambda funtion to all the items in an iterable 
l1 = [2,3,4]
map(lambda x: x ** 2, l1)
list(map(lambda x: x ** 2, l1))
# filtering the data using 'map' (it prints boolean)
l1 = [-1, -3, 4, 0, 5, -6, 6]
positive_nums = list(map(lambda x: x > 0, l1))
positive_nums
# filtering the data using 'Filter'  (it prints values)
positive_nums = list(filter(lambda x: x > 0, l1))
positive_nums
# importing numpy library
import numpy as np
# Create one dimensional array
# array(object, dtype=None, copy=True, order='K', subok=False, ndmin=0)
a1 = np.array([23, 44, 33, 66, 77, 88, 99, 12])
a1
type (a1)
# Create a two dimensional array
a2 = np.array([[23, 44, 33, 66], [77, 88, 99, 12]])
a2
a1 = np.zeros((4,3))  # Create an array of all zeros
a1
np.ones((3, 3))

np.empty((3, 3), str)
np.identity(3)
np.random.random((3, 4))
np.random.randint(50, 100, size = (2, 3))
np.arange(10)
a1 = np.array([1,5,4,3,5,4,3,2,7,6,5,4])
a1
a2 = a1.reshape(3, 4)
a2
a2.transpose()
# Create a numpy array
a = np.arange(1, 21).reshape(5, 4)
a
# Get the size of the array
a.size
# Get the shape of the array
a.shape
# No of dimensions of the array
a.ndim
# Get the data type of the elements in the array
a.dtype
# Bytes consumed by array elements
a.nbytes
a1 = np.arange(21)
a1
# a1[low, high - 1]
a1[4]
a1[6:12]
a1[12: 6:-1]
# Create a numpy array
a1 = np.arange(1, 21).reshape(5, 4)
a1
a1[2,2]
a1[2:4, 1:3]
# get the third row
a1[2]
# Get 1 and 2 row
a1[0:2]
# Boolean indexing
a1 >= 9
# Index the data  
a1[a1 >= 9]
a1 = np.random.randint(50, 100, size = (4, 6))
a1
a1[1,1] = 0
a1
a1 > 85
a1[a1 > 85] = 85
a1
a = a1.mean()
b = a1.sum()
c = a1.min()
d = a1.max()
print(a)
print(b)
print(c)
print(d)
# update the values in the array by the mean where the values are <= 60
a1 [a1 <= 60] = a1.mean()
a1
a1[(a1 <= 60) | (a1 >= 90)] = a1.mean()
a1
x = np.random.randint( 10, size = (6,9))
y = np.random.randint( 100, size = (6,2))
x
y
# combine the data
np.hstack([x, y])
# Addition, Substraction, Multiplication and other matrix operations
x = np.random.randint( 100, size = (3,3) )
y = np.random.randint( 100, size = (3,1) )
x
y
# Addition
x + y
# Subtraction
x - y
# Multiply
x * y
x = np.random.randint( 10, size = (4,4) )
x
# Get the sum, mean and median
print(x.sum(), x.mean(), x.std())
# To get summary row wise, use axis = 1
np.sum(x, axis = 1)