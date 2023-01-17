import numpy as np
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
print(np.__version__)
np.zeros(10)
np.zeros(10,dtype=int)
np.zeros((3,2))
Z = np.zeros(10)
Z
Z[4] = 1
Z
np.arange(10,20)
np.linspace(10,20,5)
Z = np.arange(20)
Z = Z[::-1]
print(Z)
np.array([1, 2, 3], dtype = complex)
x = np.array([1, 2])   # Default datatype
print(x.dtype)
x = np.array([1.0, 2.0])
print(x.dtype)
x = np.array([1, 2], dtype=np.int64)
print(x.dtype) 
np.arange(9).reshape(3,3)
np.nonzero([1,2,0,0,4,0])
np.eye(3)
Z = np.random.random((3,3,3))
print(Z)
Z = np.random.random((5,5))
Z
Z.min()
Z.max()
Z.mean()
Z = np.ones((5,5))
Z
np.pad(Z, pad_width=1, mode='constant', constant_values=0)
print(0 * np.nan)
print(np.nan == np.nan)
print(np.inf > np.nan)
print(np.nan - np.nan)
print(np.nan in set([np.nan]))
print(0.3 == 3 * 0.1)
a = np.arange(10)
a
##(start:stop:step)
a[2:7:2] 
a[4:]
a = np.array([[1,2,3],[3,4,5],[4,5,6]]) 
a 
# Returns array of items in the second column 
a[...,1]
# Will slice all items from the second row 
a[1,...] 
# Will slice all items from column 1 onwards 
a[...,1:]
# Slicing using advanced index for column
a[1:3,[1,2]]
a = np.array([1,2,3,4]) 
b = np.array([10,20,30,40]) 
a * b
# First Array + Second Array
a = np.array([[0.0,0.0,0.0],[10.0,10.0,10.0],[20.0,20.0,20.0],[30.0,30.0,30.0]]) 
b = np.array([0.0,1.0,2.0]) 
a + b
np.dot(np.ones((5,3)), np.ones((3,2)))
np.dot(np.array([[1,7],[2,4]]),np.array([[3,3],[5,2]]))
Z = np.arange(11)
Z
Z[(3 < Z) & (Z <= 8)] *= -1
Z
# Will print the items greater than 5
Z[Z > 5]
a = np.arange(0,60,5).reshape(3,4)
a
for x in np.nditer(a):
    print(x,end=' ')
# The flattened array is
a.flatten()
# Transposed from (3,4) to (4,3)
a.T
a = np.array([[1,2],[3,4]])
b = np.array([[5,6],[7,8]])
# Horizontal stacking
np.hstack((a,b))
# Vertical stacking
np.vstack((a,b)) 
a = np.array([0,30,45,60,90])
# Sine of different angles:
np.sin(a*np.pi/180)
# Tangent values for given angles:
np.tan(a*np.pi/180)
a = np.array([[3,7,5],[8,4,3],[2,4,9]])
a
np.amin(a,axis=1)
np.amin(a,axis=0)
# returns the range (maximum-minimum)
np.ptp(a,axis=1)
# Applying mean() function along axis 0
np.mean(a, axis = 0) 
yesterday = np.datetime64('today', 'D') - np.timedelta64(1, 'D')
today     = np.datetime64('today', 'D')
tomorrow  = np.datetime64('today', 'D') + np.timedelta64(1, 'D')
print(yesterday)
print(today)
print(tomorrow)
Z = np.arange('2016-07', '2016-08', dtype='datetime64[D]')
print(Z)
x = np.linspace(0,1,10,endpoint=False)[1:]
print(x)
aa = np.linalg.inv(x.reshape(3,3))
aa.dot(x.reshape(3,3))
Z = np.random.random(10)
Z[Z.argmax()] = 0
print(Z)
Z = np.arange(9).reshape(3,3)
for index, value in np.ndenumerate(Z):
    print(index, value)
X = np.random.rand(5, 5)
Y = X - X.mean(axis=1, keepdims=True)
print(Y)
A = np.arange(25).reshape(5,5)
A
A[[0,1]] = A[[1,0]]
print(A)
Z = np.random.uniform(0,1,(10,10))
U, S, V = np.linalg.svd(Z) # Singular Value Decomposition
rank = np.sum(S > 1e-10)
print(rank)
Z = np.arange(10000)
np.random.shuffle(Z)
n = 5
print (Z[np.argpartition(-Z,n)[:n]])
a = np.array([1,2,3,4,5]) 
np.save('outfile',a)
b = np.load('outfile.npy') 
b
# with txt file format
a = np.array([1,2,3,4,5]) 
np.savetxt('out.txt',a) 
b = np.loadtxt('out.txt') 
b 