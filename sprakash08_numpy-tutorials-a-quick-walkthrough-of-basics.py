#import for numpy



import numpy as np
list = [1,2,3] # Array of rank one

numpyList = np.array(list)

print("NumpyList ",numpyList)

print(type(numpyList))

print("Elements of list ", numpyList[0],numpyList[1],numpyList[2])

print("Shape of numpy list ",numpyList.shape)
numpyList[0] = 5

print("The List after replacing 1 with 5 is ", numpyList)
b = np.array([[12,13,14],[13,14,15]]) #Arry of rank-2
print("List elements are \n", b)

print("Shape of the numpy array ", b.shape)
#Zeroes

np.zeros(shape=(2,2))
## Ones

np.ones(shape=(3,3))
## Full

np.full(shape=(4,4),fill_value=10)
### Eye

np.eye(3) # giving N=3 provides a 3X3 identity matrix
## Random matrix creation using numpy



np.random.random(size=(3,3))
# Fix a seed and create a numpy matrix

np.random.seed(101)

np.random.randint(low=0,high=50,size=(10,5))
np.random.seed(123)

array = np.random.randint(low=1,high=50,size=(10,5))
array
copyOfArray = array.copy()
slice = copyOfArray[:2,1:3] # Take first to rows and fetch 1 and 2 columns of the matrix
slice
copyOfArray[:,1:2]
#Original array changed

array[:1,:] = 9999

array
copyOfArray[3:6,2:5]
# Note: Array Indexing will yield a lower rank array
copyOfArray[1,:].shape # Rank-1 array
copyOfArray[1:2,:].shape # Rank-2 array
np.random.seed(3213)

a = np.random.randint(low=1,high=50,size=(5,10))
a[a>20].reshape((6,5))
print(np.array([1,2,3]).dtype)
print(np.array([1.0,2,3.8]).dtype)
a=np.array([[1,2,3],[3,4,5]],dtype=np.float64)

print(a.dtype)
np.random.randint(312)

a = np.random.randint(low=0,high=10,size=(3,3))

b = np.random.randint(low=11,high=20,size=(3,3))

print("Matrix A")

print(a)

print()

print("Matrix B")

print(b)
# Addition of matrix

np.add(a,b)
# Sub of matrix

np.subtract(a,b)
# Multiply of matrix

np.multiply(a,b)
# Divide of matrix

np.divide(a,b)
# SQRT of matrix

np.sqrt(a)
a
np.dot(a,b)
a.dot(b)
np.sum(a)
np.sum(a,axis=0) ## Accross Column sum
np.sum(a,axis=1) ## Accross rows sum