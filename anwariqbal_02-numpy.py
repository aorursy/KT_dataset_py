

import numpy as np 

#linear algebra

import pandas as pd 

#data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
height = [1.73, 1.68, 1.71, 1.89, 1.79]

np_height = np.array(height)

print(np_height)

weight = [73.5, 64.8, 71.4, 89.5, 79.14]

np_weight = np.array(weight)

print(np_weight)

bmi = np_weight/np_height**2

print(bmi)

# List can not perform this operation based on each element.

# Numpy arrays: contain only one type

# If you try to make one, everything will be converted to strings.

#Different types: different behavior

python_list = [1, 2, 3]

print(python_list + python_list)



numpy_array = np.array([1, 2, 3])

print(numpy_array+numpy_array)

NPA = np.array([21.52, 20.45, 21.75, 24.15, 21.41])

print(NPA[3])

print(NPA>21)

print(NPA[NPA>21])
# Create list baseball

baseball = [180, 215, 210, 210, 188, 176, 209, 200]



# Import the numpy package as np

import numpy as np



# Create a numpy array from baseball: np_baseball

np_baseball = np.array(baseball)



# Print out type of np_baseball

print(type(np_baseball))
np.array([True, 1, 2]) + np.array([3, 4, False])
np_2d = np.array([[1.73, 1.68, 1.71, 1.89, 1.79],

         [73.5, 64.8, 71.4, 89.5, 79.14]])

print(np_2d)

print(np_2d.shape)

print(np_2d[0][2])

print(np_2d[0, 2])

print(np_2d[:, 1:3])

print(np_2d[1, :])
# Create baseball, a list of lists

baseball = [[180, 78.4],

            [215, 102.7],

            [210, 98.5],

            [188, 75.2]]



# Import numpy

import numpy as np



# Create a 2D numpy array from baseball: np_baseball

np_baseball = np.array(baseball)





# Print out the type of np_baseball

print(type(np_baseball))



# Print out the shape of np_baseball

print(np_baseball.shape)
print(np.mean([1, 2, 4, 6]))

print(np.median([1, 2, 4, 6]))

print(np.corrcoef())

print(std())

print(sum())

print(sort())

#Arguments for np.random.normal()

#1 distribution mean

#2 Distribution standard deviation

#3 number of samples

height = np.round(np.random.normal(1.7, 0.20, 50), 2)

print(height)

weight = np.round(np.random.normal(60.32, 5, 50), 2)

print(weight)

np_city = np.column_stack((height, weight))

print(np_city)
#Define array using numpy

x = np.array([1, 2, 4, 6], np.int16)
#Print X

x
#Print x using function

print(x)

#Print type of x

print(type(x))
#Print type of x

type(x)
#Print Index value

print(x)

print(x[0])

print(x[1])

print(x[2])

print(x[3])

print(x[-1])
#Two dimensional array

x = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], np.int16)

print(x)

type(x)
print(x[1,2])
#Slice 3rd column

print(x[:,2])

#slice first row

print(x[0,:])
print(x[1,:3])
z = np.array([[[1,2,3],[4,5,6]],[[0,-1,-2],[-3,-4,-5]]])
print(z)
print(z[1,1,0])
#slice

print(z[0:,0:,:1])
print(x.shape)

print(z.shape)



#Dimensions

print(x.ndim)

print(z.ndim)
#DTYPE

print(x.dtype)

print(z.dtype)
#SIZE

print(x.size)

print(z.size)
print(x.nbytes)

print(z.nbytes)
#Transpose

print(x.T)

print( "New line")

print(z.T)
#positive infinity

print(np.inf)



#negative infinity

print(np.NINF)



#Not a number

print(np.NAN)



#Negative zero

print(np.NZERO)



#Positive zero

print(np.PZERO)



print(np.e)

print(np.euler_gamma)

print(np.pi)
i = np.empty([3,3], np.int16)

print(i)
a = np.eye(3, dtype = np.uint8)

print(a)

a = np.eye(6, dtype = np.uint8, k=2)

print(a)
a = np.eye(6, dtype = np.uint8, k=-1)

print(a)
a = np.identity(6, dtype = np.uint8)

print(a)
a = np.ones((2, 4, 6), dtype = np.uint16)

print(a)
a = np.zeros((2, 4, 6), dtype = np.uint16)

print(a)
a = np.full((2, 4, 6), dtype = np.uint16, fill_value=3)

print(a)
a = np.tri(4, 4, k=1, dtype = np.uint16)

print(a)
a = np.tri(6, 6, k=-1, dtype = np.uint16)

print(a)
a = np.ones((5, 5), dtype = np.uint8)

b = np.tril(a, k=0)

print(b)

a = np.ones((5, 5), dtype = np.uint8)

b = np.triu(a, k=-1)

print(b)