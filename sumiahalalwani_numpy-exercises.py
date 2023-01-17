
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
print(np.__version__)





x = np.array([1, 2])   # Default datatype
print(x.dtype)
x = np.array([1.0, 2.0])
print(x.dtype)
x = np.array([1, 2], dtype=np.int64)
print(x.dtype) 






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
# write your answer here adding array a and b.
#???
# First Array + Second Array
a = np.array([[0.0,0.0,0.0],[10.0,10.0,10.0],[20.0,20.0,20.0],[30.0,30.0,30.0]]) 
b = np.array([0.0,1.0,2.0]) 
# write your answer here multiplying array a and b.
#???

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
# write your code here.
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