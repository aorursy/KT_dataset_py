import random

import math

import numpy as np



random.seed(0)

rand_list = [random.randint(0, 100) for r in range(1000)]



def list_sqrt(list):

    return [math.sqrt(n) for n in list]



%timeit rand_list_sqrt = list_sqrt(rand_list)

np_rand_array = np.array(rand_list)

%timeit np_rand_array_sqrt = np.sqrt(np_rand_array)
# From a Python list

array1 = np.array([1, 2, 3, 4, 5])

array1
# Print the underlying data type used by the array.

array1.dtype
# NumPy's equivalent of `range`

array2 = np.arange(0, 100, 2)

array2
# Using a specific data type. See the following page for a list of available types: https://numpy.org/doc/1.17/user/basics.types.html

array3 = np.arange(0, 100, 2, dtype='float32')

array3
# Fill an array with zeros

array4 = np.zeros(10, dtype='float64')

array4
# Or ones

array5 = np.ones(10, dtype='int32')

array5
# Or custom number

array6 = np.full(30, 2.7818)

array6
# Or a range between certain numbers, with a certain stride (3 in this case).

array7 = np.arange(0, 100, 3)

array7
# Or a range between certain numbers, specifying the total number of numbers instead of stride.

array8 = np.linspace(0, 10, 21) # 21 uniformly-spaced numbers between 0 and 10

array8
# Randomly generating an array of floats.

array9 = np.random.random(10)

array9
# Or an array of integers between a specific range

array10 = np.random.randint(10, 20, size=50) # 50 random integers between 10 and 20.

array10

# Creating a 2-D array of zeros.

array11 = np.zeros((10, 10))

array11
# Similarly, the type can be specified.

array12 = np.zeros((10, 10), dtype='int64')

array12
# Or fill with ones.

array13 = np.ones((5, 5), dtype='int64')

array13
# Creating 3-D arrays is as easy.

array14 = np.ones((3, 3, 3), dtype='int64')

array14
# Back to 2-D arrays, you could also convert a Python list of lists into an array.

array15 = np.array([[1, 2, 3],

                    [2, 3, 2],

                    [3, 2, 1]])

array15

array16 = np.ones((3, 5), dtype='float64')

print(array16.ndim)

print(array16.shape)

print(array16.dtype)

print(array16.itemsize)

print(array16.size)

# I will use this array for the examples of this section.

np.random.seed(0)

array17 = np.random.randint(0, 10, size=(4, 5))

array17
# Extract the first row.

array17[0, :]
# Or the last row.

array17[-1, :]
# Extract the first and third columns.

array17[:, [0, 2]]
# Extract the element in the 3rd row and 4th column. This should have been the first example, shouldn't it?

array17[2, 3]
# From the first row, extract the third element up until the end.

array17[0, 2:]
# Extract the elements in the middle.

array17[1:3, 1:4]
# Like indexing for Python lists, you could use negative strides to arrays. The line below extract the first row in reversed order.

array17[0, ::-1]
array18 = array17.copy() # Copy the array so we don't modify the original one.

array18
array18[0, 0] = 100

array18
# Change the first three elements of the first row in one operation.

array18[0, 0:3] = (1000, 2000, 3000)

array18
# Change the first and third elements of the fourth column.

array18[[0, 2], 3] = [5000, 6000]

array18
array19 = np.arange(0, 10)

array19
# Find the squares of the numbers.

array20 = np.square(array19)

array20
# Let's find the difference between the numbers and their squares.

array21 = np.subtract(array20, array19)

array21
# Actually, for substraction and other elementary functions, you could simply

# use the Python operators, but I wanted to illustrate the original function

# which the operator will end up calling.

array21 = array20 - array19

array21

# Find the sines for values between 0 and 2*pi.

array22 = np.linspace(0, 2*np.pi, 20)

array22_sin = np.sin(array22)

array22_sin
# and the cosines

array22_cos = np.cos(array22)

array22_cos
# Having the sines and cosines, we might as well draw the circle!

from matplotlib import pyplot as plt

plt.figure(figsize=(4, 4))

plt.plot(array22_sin, array22_cos)
# Let's use NumPy to sum the numbers from 1 to 100 to see whether Gauss was right: https://en.wikipedia.org/wiki/Carl_Friedrich_Gauss#Anecdotes

array23 = np.arange(1, 101)

np.sum(array23)
# To find the average:

np.average(array23)
# To find the minimum:

np.min(array23)
# To find the maximum:

np.max(array23)
array24_x = np.random.randint(-10, 10, size=50) # 50 random integers between 10 and 20.

array24_y = np.random.randint(-10, 10, size=50) # 50 random integers between 10 and 20.

array24_x_centre = np.average(array24_x)

array24_y_centre = np.average(array24_y)



from matplotlib import pyplot as plt

plt.figure(figsize=(6, 6))

plt.plot(array24_x, array24_y, 'o', color='green')

plt.plot(array24_x_centre, array24_y_centre, 'x', color='red')

array24 = np.random.randint(0, 100, size=50)

array24
# Sort the array.

array25 = np.sort(array24)

array25
array26 = np.random.randint(10, 30, size=50) # random temperatures in Celesius

array26
# Comparison operators produce a True/False array.

array26 > 20
# By passing in a True/False array to another array, we could extract

# the elements for which we pass True.

array26[array26 > 20]
# We could then use a function like count_nonzero on the boolean array

# to find the number of days in whech the temparuter is above 20.

np.count_nonzero(array26 > 20)
# Let's start by generating an array of 10 random numbers between 0 and 100.

array27 = np.random.random(10)*100

array27
# Find the average

array27_mean = np.average(array27)

array27_std = np.std(array27)

print(f"Mean = {array27_mean}")

print(f"Standard Deviation = {array27_std}")

# Normalize the array: substract the average from all the elements and divide the

array27_normal = (array27 - array27_mean)/array27_std

array27_normal
 # reduce precision to make printing compact

%precision 2

array28 = np.random.random((3, 7))*100

array28
# Assuming each column represent a sample, find the mean of each column.

array28_mean = np.average(array28, axis=0) # Notice that we have to specify the dimension,

                                           # otherwise the function would average all elements.

array28_mean



# Similarly, find the standard deviation.

array28_std = np.std(array28, axis=0)

array28_std
# Now let's do only the substraction so we could see what is happening

array28 - array28_mean
# Now find the full normalization

(array28 - array28_mean)/array28_std