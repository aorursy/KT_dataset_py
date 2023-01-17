# Once you've installed NumPy you can import it as a library:

import numpy as np
my_list = [1,2,3]

my_list
np.array(my_list)
my_matrix = [[1,2,3],[4,5,6],[7,8,9]]

my_matrix
np.array(my_matrix)
np.arange(0,10)
np.arange(0,11,2)
np.zeros(3)
np.zeros((5,5))
np.ones(3)
np.ones((3,3))
np.linspace(0,10,3)
np.linspace(0,10,50)
np.eye(4)
np.random.rand(2)
np.random.rand(5,5)
np.random.randn(2)
np.random.randn(5,5)
np.random.randint(1,100)
np.random.randint(1,100,10)
arr = np.arange(25)

ranarr = np.random.randint(0,50,10)
arr
ranarr
arr.reshape(5,5)
ranarr
ranarr.max()
ranarr.argmax()
ranarr.min()
ranarr.argmin()
# Vector

arr.shape
# Notice the two sets of brackets

arr.reshape(1,25)
arr.reshape(1,25).shape
arr.reshape(25,1)
arr.reshape(25,1).shape
arr.dtype
import numpy as np
#Creating sample array

arr = np.arange(0,11)
#Show

arr
#Get a value at an index

arr[8]
#Get values in a range

arr[1:5]
#Get values in a range

arr[0:5]
#Setting a value with index range (Broadcasting)

arr[0:5]=100



#Show

arr
# Reset array, we'll see why I had to reset in  a moment

arr = np.arange(0,11)



#Show

arr
#Important notes on Slices

slice_of_arr = arr[0:6]



#Show slice

slice_of_arr
#Change Slice

slice_of_arr[:]=99



#Show Slice again

slice_of_arr
#Now note the changes also occur in our original array!
arr
#Data is not copied, it's a view of the original array! This avoids memory problems!
#To get a copy, need to be explicit

arr_copy = arr.copy()



arr_copy
arr_2d = np.array(([5,10,15],[20,25,30],[35,40,45]))



#Show

arr_2d
#Indexing row

arr_2d[1]

# Format is arr_2d[row][col] or arr_2d[row,col]



# Getting individual element value

arr_2d[1][0]
# Getting individual element value

arr_2d[1,0]
# 2D array slicing



#Shape (2,2) from top right corner

arr_2d[:2,1:]
#Shape bottom row

arr_2d[2]
#Shape bottom row

arr_2d[2,:]
#Set up matrix

arr2d = np.zeros((10,10))
#Length of array

arr_length = arr2d.shape[1]
#Set up array



for i in range(arr_length):

    arr2d[i] = i

    

arr2d
#Fancy indexing allows the following
arr2d[[2,4,6,8]]
#Allows in any order

arr2d[[6,4,2,7]]
arr = np.arange(1,11)

arr
arr > 4
bool_arr = arr>4
bool_arr
arr[bool_arr]
arr[arr>2]
x = 2

arr[arr>x]
import numpy as np

arr = np.arange(0,10)
arr + arr
arr * arr
arr - arr
# Warning on division by zero, but not an error!

# Just replaced with nan

arr/arr
# Also warning, but not an error instead infinity

1/arr
arr**3
#Taking Square Roots

np.sqrt(arr)
#Calcualting exponential (e^)

np.exp(arr)
np.max(arr) #same as arr.max()
np.sin(arr)
np.log(arr)