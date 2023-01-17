import numpy as np

import time



list1 = []

list2 = []



for i in range(1000):

    list1.append(i)

print(len(list1))



for j in range(1000,0,-1):

    list2.append(j)

print(len(list2))



result = []

start = time.time()

for i in range(len(list1)):

    result.append(list1[i]+list2[i])

end = time.time()

print("Time taken in python list",(end - start)*100000)



a = np.array(list1)

b = np.array(list2)



start1 = time.time()

result = a + b

end1 = time.time()

print("Time taken in numpy list",(end1 - start1)*100000)
one_d_array = np.array([1,2,3])

print(one_d_array)

two_d_array = np.array([[1.0,2.0,3.0],[4.0,5.0,6.0]])

print(two_d_array)
print(one_d_array.ndim)

print(two_d_array.ndim)

print(one_d_array.shape)

print(two_d_array.shape)

print(one_d_array.dtype)

print(two_d_array.dtype)

# To specify the data type while initializing

one_d_array = np.array([1,2,3],dtype = 'int16')

print(one_d_array.dtype)

print(one_d_array.size)

print(two_d_array.size)

# item Size of an array

print(one_d_array.itemsize)

print(two_d_array.itemsize)

print(one_d_array.size * one_d_array.itemsize)

print(two_d_array.size * two_d_array.itemsize)

print(one_d_array.nbytes)

print(two_d_array.nbytes)

c = np.array( [ [1,2], [3,4] ], dtype=complex )

print(c)
array = np.array([[1,3,5,7],[8,10,12,14]])

# Get a specific element [r, c]

array[1, 2]

# Get a specific row ( 0 denotes first row and all column values )

array[0, :]

# Get a specific column ( 0 denotes first column and all row values )

array[:, 0]

array[1, 3] = 12

array[0, :] = [1,2,3,4]

three_d_array = np.array([[[1,2],[2,4],[3,6]]])

three_d_array[0,1,1] = 7

array = np.array([[1,3,5,7],[8,10,12,14]])



# All 0s matrix

print(np.zeros((2,2)))



# All 1s matrix

print(np.ones((2,2,2), dtype='int32'))



# other number

print(np.full((2,2), 8))



# Any other number (full_like) of other shape

print(np.full_like(array, 4))



#empty array

print(np.empty( (2,3) ) )





# Random decimal numbers

print(np.random.rand(3,4))



# Random Integer values

print(np.random.randint(2,8, size=(3,3)))



# identity matrix

print(np.identity(3))



# Repeat an array

temp = np.array([[1,2,3,4,5,6]])

print(np.repeat(temp,3, axis=0))



# Repeat an array

temp = np.array([[1,2,3,4,5,6]])

print(np.repeat(temp,3, axis=1))
arr = np.array([1,2,3])

temp = arr

temp[0] = 5

print(temp)

print(arr)

arr = np.array([1,2,3])

temp = arr.copy()

temp[0] = 5

print(temp)

print(arr)

a = np.array([1,2,3,4])

print(a)



# Element wise

print(a+2)

print(a-2)

print(a*2)

print(a/2)

print(a+a)

print(np.exp(2))

print(a**2)

print(np.cos(a))

a = np.random.randint(2,8, size=(3,3))

b = np.random.randint(2,8, size=(3,3))

print(a,b)



print(np.linalg.solve(a, b))



print(np.linalg.inv(a))



print(np.trace(a))



print(np.linalg.norm(a))



print(np.linalg.eigvals(a))



print(np.dot(a,b))
initial = np.array([[10,20,30,40],[50,60,70,80]])

print(initial)



final = initial.reshape((2,4))

print(final)



print(np.ravel(initial))



print(np.transpose(initial))



print(np.column_stack((initial,final)))



# Vertically stacking vectors

v1 = np.array([1,2,3,4])

v2 = np.array([5,6,7,8])



print(np.vstack([v1,v2,v1,v2]))



# Horizontal  stack

h1 = np.ones((2,4))

h2 = np.zeros((2,2))



print(np.hstack((h1,h2)))