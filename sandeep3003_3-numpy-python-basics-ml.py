# using numpy

import numpy as np
# Creating a python list

my_list = [1,2,3]

# converting list to numpy array

np.array(my_list)
# creating 2d list

my_matrix = [[1,2,3],[4,5,6],[7,8,9]]

# converting into numpy 2d array

np.array(my_matrix)
# create an numpy array (from, until)

np.arange(0,10)
# create an numpy array with (from, until, difference)

np.arange(0,10,2)
# creating numpy array of zeros

np.zeros(3)
# creating a 2d numpy zeros array

np.zeros((5,5))
# creating a numpy array of ones

np.ones(3)
# creating a 2d numpy array of ones



np.ones((3,3))
# create a numpy array of (from, untill, terms)

np.linspace(0,10,3)
np.linspace(0,10,50)
np.random.rand(2)
# create a 2d numpu array with random integer

np.random.rand(5,5)
np.random.randn(2)
np.random.randn(5,5)
np.random.randint(1,100)
# create a random array with (from, untill, terms)

np.random.randint(1,100,10)
ranarr = np.random.randint(0,50,10)

ranarr
# finding the max of numpy array

ranarr.max()
# finding the index of max of numpy array

ranarr.argmax()
# finding the min of numpy array

ranarr.min()
# finding the index of min of numpy array

ranarr.argmin()
arr = np.arange(0,11)

arr
arr.dtype
# Get a value at an index

arr[8]
# Get values in a range

arr[1:5]
# Get values in a range

arr[0:5]
# Setting a value with index range (Broadcasting)

arr[0:5]=100

arr
# Reset array

arr = np.arange(0,11)
# Important notes on Slices

slice_of_arr = arr[0:6]

slice_of_arr
# Change Slice

slice_of_arr[:]=99

slice_of_arr
arr
# To get a copy, need to be explicit

arr_copy = arr.copy()

arr_copy
# indexing 2d array

arr_2d = np.array(([5,10,15],[20,25,30],[35,40,45]))

arr_2d
# Indexing row

arr_2d[1]
# Format is arr_2d[row][col] or arr_2d[row,col]

# Getting individual element value

arr_2d[1][0]
# Getting individual element value

arr_2d[1,0]
# 2D array slicing

# Shape (2,2) from top right corner

arr_2d[:2,1:]
# Shape bottom row

arr_2d[2]
# Shape bottom row

arr_2d[2,:]
arr = np.arange(1,11)

arr
# returns a boolean of array with the satisfied condition

arr > 4
# assigning the boolean array

bool_arr = arr>4

bool_arr
# selecting the value for which the condition is satisfied

arr[bool_arr]
arr[arr>2]
x = 2

arr[arr>x]
arr
# adding two numpy arrays

arr + arr
# multiplying two arrays

arr * arr
# substracting two arrays

arr - arr
# divison of two arrays

arr/arr
1/arr
# exponential of the array

arr**3
# Taking Square Roots

np.sqrt(arr)
# Calcualting exponential (e^)

np.exp(arr)
np.max(arr) #same as arr.max()
# sin of the numopy array

np.sin(arr)
# log of a numpy array

np.log(arr)