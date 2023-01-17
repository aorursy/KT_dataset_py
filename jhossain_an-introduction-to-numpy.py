# add 2 lists 
L1 = [1, 2, 3]
L2 = [4, 5, 6]
print(L1+L2)
# element wise sum using numpy array 
import numpy as np 
A1 = np.array([1, 2, 3])
A2 = np.array([4, 5, 6])
print(A1+A2)
import numpy as np
import time
import sys
S = range(1000)
print("Python List: ", sys.getsizeof(5)*len(S))
 
D = np.arange(1000)
print("Numpy Array: ", D.size*D.itemsize)
import time
import sys
 
SIZE = 1000000
 
L1 = range(SIZE)
L2 = range(SIZE)
A1 = np.arange(SIZE)
A2 = np.arange(SIZE)
 
start= time.time()
result=[(x,y) for x,y in zip(L1,L2)]
# time in ms 
print((time.time()-start)*1000)
 
start = time.time()
result = A1+A2
# time in ms 
print((time.time()-start)*1000)
%timeit sum(range(1000))
%timeit np.sum(np.arange(1000))
# import numpy 
import numpy as np 
# Creating 1D array
A = np.array([1, 2, 3])
print(A)
# type 
print(type(A))
# create an array with categorical entities. 
X = np.array([12, 13, "n"])
print(X)
# type 
print(type(X))
# Creating 2D array
A2 = np.array([[3, 4, 5], [7, 8, 9]])
print(A2) 
# Creating 3D array
A3 = np.array([[(1, 2, 3), (4, 5, 6)], [(7, 8, 9), (10, 11, 12)]])
print(A3) 
A1 = np.array([1, 2, 3,4, 5])
# size 
A1.size
A2 = np.array([[4, 5, 6], [7, 8, 9]])
# shape 
A2.shape 
# get row 
A2.shape[0]
# get column
A2.shape[1]
A3 = np.linspace(0, 100, 6)
# dtypes 
A3.dtype
A4 = np.ones((2,3))
# convert 
A4.astype(np.float16)
A5 = np.linspace(0, 100, 20)
# array to list 
A5.tolist() 
np.info(np.linspace)