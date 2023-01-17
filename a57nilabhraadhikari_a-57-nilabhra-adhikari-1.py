#Creation of 1D array

import numpy as np

arr1=np.array([1,2,3,4,5,6,7,8,9])

print(arr1)
#Creation of Boolean Array

import numpy as np

arr1=np.array([1,2,0,True,False],dtype=np.bool)

arr1
#Creation of 2D array

import numpy as np

l=[[1,2,3],[4,5,6],[7,8,9]]

a=np.array(l)

print(a)
import numpy as np

a=np.array([[1,2,3],[4,5,6],[7,8,9]])

print(a[a%2==1])
import numpy as np

array1 = np.array([0, 10, 20, 40, 60])

print("Array1: ",array1)

array2 = [10, 30, 40]

print("Array2: ",array2)

print("Common values between two arrays:")

print(np.intersect1d(array1, array2))
import numpy as np

a=np.array([[1,2,3],[4,5,6],[7,8,9]])

a[a%2==0]=0

print(a)

import numpy as np

a=np.array([33,33,13,44,55,66,77,55,12,23,21,34,59])

b=np.array([21,34,55,77])

x=b.argsort()

out=a[b[x[np.searchsorted(b,a,sorter=x)]]!=a]

print(out)
import numpy as np

a=np.array([1,5,8,3,6,9])

b=np.array([2,4,6,8,6,5])

result=np.where(a==b)

print(result)