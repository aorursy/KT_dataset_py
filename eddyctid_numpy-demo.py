import pandas as pd
import numpy as np
arr = np.array([[1,2,3],[4,5,6],[7,8,9]]) # array creation
type(arr) #ndim array
arr.ndim # no of dimension
arr.shape # tuple matrix
arr.size # no of items
arr.dtype # data type
arr.itemsize #bytesize
# Creating Array by defining Data Type
newArr = np.array([[1,2,3],[4,5,6],[7,8,9]],dtype='float64')
newArr
newArr.dtype
# Zeros, Ones are special functions &  have default float64 datatype but datatype can be specified  explicitly
np.zeros([3,3])
np.zeros([3,3],dtype='int32')
np.ones([3,3])
np.ones([3,3],dtype='complex')
#empty function creates array with uninitialized values
np.empty([2,3])
# NumPY sequence arrays using 'range'
np.arange(20)
np.arange(0,51,5)# Start, Stop, Interval
np.arange(10,20,0.5)
np.linspace(10,20,20)# Start, Stop, No of Elements needed
from numpy import pi
np.sin(np.linspace(0,2*pi,10))
#reshape
arr
arr.reshape([9,1])
np.arange(10).reshape([5,2])
# Matrix Operations in np.array
np.arange(4)**2
np.arange(10)+np.linspace(10,30,10)
np.arange(10) < 7 # boolean ops
(np.arange(9).reshape([3,3]))*(np.arange(0,9,1).reshape([3,3])) # Element wise product
(np.arange(9).reshape([3,3])).dot(np.arange(0,9,1).reshape([3,3])) # Matrix Multiplication
#random function is used for test arrays
testArr = np.random.random([3,3])
testArr
testArr*=3
testArr

