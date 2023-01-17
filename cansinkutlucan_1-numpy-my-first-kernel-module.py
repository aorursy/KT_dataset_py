# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Numpy is used to work with arrays

# Let's create some arrays

array1 = np.array([1,2,3,4,5,6,7,8])                   # 1 dimensional array

array2 = np.array([2,3,4,5,6,7,8,9])                   # 1 dimensional array

array3 = np.array([[1,2,3,4],[5,6,7.5,8],[9,10,11,12]])  # 2 dimensional array

array4 = np.array([[2,3,4,5],[6,7,8.5,9],[10,11,12,13]]) # 2 dimensional array
# Our array1 and array2 are 1 dimensional, array3 and array4 are 2 dimensional.

# Well, We can start working out the details for array1 and array3.

print("dimension of array1:",array1.ndim)

print("dimension of array3:",array3.ndim)

print("shape of array1:",array1.shape)

print("shape of array3:",array3.shape)

print("size of array1:",array1.size)

print("size of array3:",array3.size)

print("datatype of array1 elements:",array1.dtype)

print("datatype of array3 elements:",array3.dtype)

print("type of array1:",type(array1))

print("type of array3:",type(array3))

# Mathematical Operations - Part 1

print(array1+array2)      # addition

print(array3+array4)

print("-"*50,"1")

print(array1-array2)      # subtraction

print(array3-array4)

print("-"*50,"2")

print(array1*array2)      # multiplication

print(array3*array4)

print("-"*50,"3")

print(array1.dot(array2)) # matrix multiplication 

array5 = np.array([[4,5,6],[7,8,9],[10,11,12],[13,14,15]])

print(array3.dot(array5))



print("-"*100,"Part 2")

print("")

# Mathematical Operations - Part 2

print("square of array1:",array1**2)

print("square of array3:",array3**2)

print(np.sin(array1))     # the term "sin" means "sine"

print(np.sin(array3))

print(array3<7)           # boolean
# Let's create a random array

array6= np.random.random((2,2))

print(array6)

print("Sum of all elements in matrix:",np.sum(array6)) 

print("Max element:",np.max(array6))

print("Min element:",np.min(array6)) 

print("Sum of rows:",array6.sum(axis=1))

print("Sum of columns:",array6.sum(axis=0))

print(np.add(array6,array6)) # Sum of itself

print("-"*50)

print(np.square(array6))     # Square

print(np.sqrt(array6))       # Square root
# Some different ways of create an array

zeros = np.zeros((2,2))

print(zeros)

zeros[0,1]=3

print(zeros)

print("-"*50)

ones = np.ones((2,2))

print(ones)

empty = np.empty((2,3))

print(empty)

print("-"*50)

print(np.arange(5,37,4))   # constant increase 4 by 4 between 5 and 37(excluding 37)

print(np.linspace(5,37,4)) # puts 4 elements between 5 and 37(including 5 and 37)
# Some basic operations for 1 dimensional array

print(array1)

array1[3] = 99     # transformation of 3rd index

print(array1)

print(array1[2:5]) # print 2nd index to 5th index  

print(array1[-1])  # print last element

print(array1[::-1]) # reverse of array1

print("-"*50)

# Some basic operations for 1 dimensional array

print(array3)

print(array3[1,1])

array3[2,1]=88

print(array3)

print("all 2nd index column:",array3[:,2])

print("all 1st index row",array3[1,:])

print(array3[:,0:2]) # print all from 0th to 2nd index(not including 2nd column

print("Last column:",array3[:,-1])

print(array3[1:3,1:3])
print(array2)

print(array4)

a=array2.ravel() #  contiguous flattened array2

b=array4.ravel() #  contiguous flattened array4

print(a)

print(b)

c=b.reshape(12,1) # reshaped array b (Temporarily)

print(c)          # NOTE: Shape of c has changed

print(b)          # NOTE: Shape of b hasn't changed

d=b.resize(2,6)   # reshaped array b (Permanently)

print(b)          # NOTE: Shape of b has changed

print(array4.T)   # Transpose of array4

# Concatenating numpy arrays

array7 = np.array([[4,8],[12,16]])

array8 = np.array([[3,7],[11,15]])

concatv = np.vstack((array7,array8)) # Vertical

concath = np.hstack((array7,array8)) # Horizontal

print(concatv)

print(concath)
# Transformation list to array

list1 = [1,2,3,4]

array9 = np.array(list1)

print(array9)

print(type(array9))

# Transformation array to list

array10 = np.array([5,6,7,8])

list2 = list(array10)

print(list2)

print(type(list2))
# Copy of arrays in memory

array11 = np.array([6,7,8,9])

# Let's say x equals array10 and y equals x, and z equals y

x=array11

y=x

z=y

# what if I change one element of y

y[2] = 77

print(x)

print(y)

print(z)

# All changed .

print("-"*50)

array12= np.array([5,6,7,8])

k=array12.copy()

l=k.copy()

m=l.copy()

l[1]=66

print(k)

print(l)

print(m)

# see the difference