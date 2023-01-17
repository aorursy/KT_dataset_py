#importing numpy 



import numpy as np # linear algebra
#initiating numpy arrays

a = np.array([1,2,3])

b = np.array([[1.3,4.5],[2.3,6.7]])

print(a)

print(b)
#Looking at the dimensions

print(a.ndim)

print(b.ndim)
#Getting shape

print(a.shape)

print(b.shape)
#Getting the type

print(a.dtype)

print(b.dtype)
#get size of each element and overall size

print(a.itemsize, a.nbytes)

print(b.itemsize, b.nbytes)
#All 0 matrix

np.zeros((3,3))
# All 1 matrix

np.ones((3,4))
# Any other number

np.full((3,3),12)
#Random Decimal Numbers

np.random.rand(2,5)
#Random Integer Numbers

np.random.randint(0,9,size = (2,3))
#Identity Matrix

np.identity(4)
#Repeating an array

a = np.array([[0,5,9]])

np.repeat(a,5,axis = 0)
##################

11111

10001

10901

10001

11111

##################

first = np.ones((5,5),dtype = "int8")

second = np.zeros((3,3),dtype = "int8")

second[1,1] = 9

first[1:4,1:4] = second

first
#Finding Sine

a = np.array([1,2,3,4,5])

np.sin(a)
#Matrix Multiplication



a = np.array([[1,2,3],[4,5,6]])

b = np.array([[1,2],[3,4],[5,6]])

np.matmul(a,b)
# Min 

a = np.array([[1,2,3],[4,5,6]])

print(np.min(a))

print(np.min(a,axis = 0))
x = np.array([1,2,3,4,5,6,np.nan])

print(np.min(x))

print(np.nanmin(x))
#Range

a = np.array([1,2,3,4,5,7,9])

b = np.array([[1,10],[99,0]])

print(np.ptp(a))

print(np.ptp(b, axis = 0))
#Percentiles

np.percentile(a,50)
#Correlation

a = np.array([1,2,3,4])

b = np.array([7,8,9,1])

np.corrcoef(a,b)
a = np.array([[1,2,3,4],[5,6,7,8]])

print(a)

print(a.reshape(1,8))



print(a.reshape(4,2))



print(a.reshape(2,2,2))
# Vertical and Horizontal Stacking

a = np.array([1,2,3,4])

b = np.array([5,6,7,8])

print(np.vstack([a,b,a,b]))

print(np.hstack([a,b,a,b]))
# Finding Unique elements

np.unique([1, 1, 2, 2, 3, 3])
a = np.array([1,2,3,4])

b = np.array([5,6,7,4])

print(np.intersect1d(a,b))

print(np.union1d(a,b))

print(np.setdiff1d(a,b))

print(np.setxor1d(a,b))