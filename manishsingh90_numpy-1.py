# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
a = np.array([1, 2, 3])

print(type(a))

print(a.shape)

print(a[0], a[1], a[2])

a[0] = 5

print(a)

np.ones((3,1))
np.random.random()
print(a.sum())

print(a.min())

print(a.max())
a.mean()
b = np.array([[1, 2, 3], [4, 5, 6]])

print(b.shape)

print(b[0, 0], b[0, 1], b[1, 0])

print(b)
#transpose of 3,2 matrix

b.T
len([[1,2],[3,4],[5,6]])

c = np.array([[1,2],[3,4],[5,6]])

print(c)

print(c.ndim)

print(c.dtype)

print(c.size)

print(c.shape)



#create array with datatype

c = np.array([1,2,3,4,], dtype=np.float64)

print(c)

print(c.dtype)

print(c.itemsize)



c = np.array([1,2,3,4,], dtype=np.int32)

print(c)

print(c.ndim)

print(c.dtype)

print(c.itemsize) #4 bytes



c = np.array([1,2,3,4,], dtype=np.int16)

print(c.itemsize) #2 bytes



c = np.array([1,2,3,4,], dtype=np.int8)

print(c.itemsize) #1 bytes



# Note: 8 = 1 bytes, 16 = 2 bytes, 32 = 4 bytes, 64 = 8 bytes

# we dont have less than int8 or float8 (1 bytes)
# ndarray.ndim

arr1  = np.array([1,2,3,4])

arr2  = np.array([[1,2,3],[4,5,6]])



print(arr1)

print(arr1.ndim)



print(arr2)

print(arr2.ndim)
#ndarray.shape

print(arr1.shape)

print(arr2.shape)
# ndarray.size; product of shape

print(arr1.size)

print(arr2.size)
#return type of element of array

arr3 = np.array([[1.0,2.1,3.1],[4.1,5.2,6.0]])

arr3_1 = np.array([[1.0,2.1,3.1],[4.1,5.2,6.0]], dtype = np.float32)





print(arr1.dtype)

print(arr2.dtype)

print(arr3.dtype)

print(arr3_1.dtype)
#Return size in byte of each element of array (dtype/8, 64/8, 32/8 bytes)

print(arr1.itemsize)

print(arr3.itemsize)

print(arr3_1.itemsize)
# ndarray.data : 

print(arr1.data)

print(arr2.data)



#

arr1  = np.array([1,2,3,4])

arr1_1  = np.array([1.0,2.1,3.1])



arr2  = np.array([[1,2,3],[4,5,6]], dtype = np.int32)

arr3  = np.array([[1,2,3],[4,5,6]])

arr4 = np.array([(1.5,2,3), (4,5,6)])

arr5 = np.array( [ [1,2], [3,4] ], dtype=complex )

# The function zeros creates an array full of zeros, 

# the function ones creates an array full of ones, and the function empty creates an array 

# whose initial content is random and depends on the state of the memory. 

# By default, the dtype of the created array is float64.

print(np.zeros( (3,4) ))

print("=====2D=== ones===")

print(np.ones((2,3), dtype=np.int16))

print("=====3D======")

print(np.ones((2,3,4), dtype=np.int16))

print("=====empty=======")

print(np.empty( (2,3) ))

print("======full========")

print(np.full([2,3], "MA"))

print(np.full([3], 2))

print(np.full([3,3], 10, dtype= float))

print("========arange======")

print(np.arange(10))

print("==============")

print(np.arange( 10, 30, 5 ))

print("==============")

print(np.arange( 0, 2, .5 ))

print("======linspace========")

print(np.linspace( 0, 2, 9 )) # 9 numbers from 0 to 2

print(np.linspace( 0, 2 * 3.14, 100))

print("======random.rand(tuple)========")

print(np.random.rand(3,2))

print("======random.randint(low,high,size=tuple)========")

print(np.random.randint(5, size=(2,3)))

print(np.random.randint(5, size=[2,3]))#will work

print(np.random.randint(5, size=(5)))

print(np.random.randint(5, 10, size=(5)))

      

print("======random.random_sample(size=tuple)========")

print(np.random.random_sample(size=4))

print(np.random.random_sample(size=(4,4)))

print(np.random.random_sample((4,4)))

print(np.random.random_sample(size=[4,2]))#will work





# Refer: array, zeros, zeros_like, ones, ones_like, empty, empty_like, arange, 

# linspace, numpy.random.RandomState.rand, numpy.random.RandomState.randn, fromfunction, fromfile



print("========zeros_like===================")

x = np.arange(6)

print(np.zeros_like(x))

# with shape

print(np.zeros_like(x.reshape(2,3)))



print(np.zeros_like(x,dtype=np.float))

print(np.zeros_like(x,dtype=float))



print("=======Ones_like====================")

print(np.ones_like(x))

print(np.ones_like(x, dtype = float))

print(np.ones_like(x.reshape(2,3), dtype = float))



print("=======empty_like====================")

print(np.empty_like(x))

print("==Int==")

print(np.empty_like(x, dtype = int))

print("==float==")

print(np.empty_like(x.reshape(2,3), dtype=float))

print(np.empty_like(x, dtype = float))



print("=========full_like(shape_like, value, dtype)=================")

print(np.full_like(x, 1))

print(np.full_like(x.reshape(2,3), 2))

print(np.full_like(x, 10, dtype= float))

print(np.full_like([1,2,3,4], 8, dtype = float))

#practising it

arr1.ndim

arr2.shape

arr2.size

arr2.dtype

arr2.itemsize



a = np.array([1,2,3,4])

np.zeros((2,2), dtype=int)

np.ones((2,2), dtype=float)

np.empty((3,3))

np.arange(10)

np.arange(10, dtype=float)

np.arange(5, 10, dtype=float)

np.arange(5, 10, 2, dtype=float)

np.linspace(5, 10, 10, dtype=float)

np.full((2,2), 10)

np.full((2,3,1), 10, dtype=float)

np.full_like(np.linspace(5, 10, 10, dtype=float), 3, dtype=float)



np.random.rand(2,2)#defouylt type is float

np.random.randint(5, 10, size=(5))

np.random.random_sample(size=(5,5))

np.full([2,3], "MA")

np.full([2,3], "2").astype('int')

np.full([2,3], 2.2).astype('int')

np.array([2.2, 3.1, 2]).astype('int')

np.array([2.2, 3, 2. ]).astype('int')

np.array([2, 3, '2' ]).astype('int')

np.array([2.5, 3, 2.1 ]).astype('str')

np.array([2.5, 3, '2.1' ]).astype('float')



# Create a boolean array

arr2d_b = np.array([1, 0, 10], dtype='bool')

arr2d_b

# Create an object array to hold numbers as well as strings

arr1d_obj = np.array([1, -1.2 ,'a'], dtype=object)

arr1d_obj
# Convert an array back to a list

print(arr1d_obj.tolist())

print(arr2)

print(arr2.tolist())
np.arange(12)**2       