# Q1 Create a Numpy Array
import numpy as np

A = np.array([1,2,3,4])
print(A)
#Q1 Creating a 2D Numpy Array
import numpy as np

A = np.arange(20).reshape(4,5)
print(A)
# Q1 Creating Boolean Array using Numpy:

import numpy as np

A = np.array([1,0.2,0,False,True,'s',None],dtype=bool)
print(A)
# Q2 Exracting Odd Numbers from a 2D Array

import numpy as np
A = np.array([[1,2,3],[4,5,6],[7,8,9]],np.int32)
answer = (A[A%2!=0])
print(answer)
#Q3 Replacing items that satisfy a condition with another value in the NumPy Array

import numpy as np

A = np.random.randint(0,5,size=(5,4))
print("a\n")
print(A)
B = (A<3).astype(int)
print("\nb")
print(B)
# Q4 Getting the common items between 2 Python Numpy Arrays

import numpy as np
A = np.array([1,2,3,4])
print("Array 1: ",A)
B = [3,4,5,6]
print("Array 2: ",B)
print("Common values between two arrays:")
print(np.intersect1d(A,B))

# Q5 Removing from one array those elements which occur in another array

import numpy as np

X = np.array([3,4,5])
print("Array to remove: ",X)
A = [1,2,3,4,5,6,7,8]
print("Initial Array: ",A)
for i in  X:
    if i in A:
        A.remove(i)
print("Array, after removal: ",A)
#Q6 Getting positions of the Elements where elements of 2 Arrays match

import numpy as np

a = np.array([0,1,2,3,4,5,6])
b = np.array([6,5,4,3,2,1,6])
np.where(a==b)
