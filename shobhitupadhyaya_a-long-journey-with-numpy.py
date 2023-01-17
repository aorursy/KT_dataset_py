import numpy as np # linear algebra
arr = np.array([1,2,3])

print(type(arr))

print(arr)

print("Elemment type: ",arr.dtype)
print("Number of dimensions:", arr.ndim)
print("Shape of an array:",arr.shape)
print("Size of an array:",arr.size)
matrix = np.array([[1,2,3], [4,5,6]])

print(matrix)
print("Element type:",matrix.dtype)

print("Number of dimension:",matrix.ndim)

print("Shape of matrix:",matrix.shape)

print("Size of matrix:",matrix.size)
f_matrix = np.array([[1.0,2.0], [3.0, 4.0]])

print(f_matrix.dtype)
c_matrix = np.array([['a','b'], ['c', 'd']])

print(c_matrix.dtype.name)
s_matrix = np.array([["hello", "Hi"], ["Google", "Facebook"]])

print(s_matrix.dtype.name)
mix = [1, 2.0]

print(mix)
print(np.array(mix))
mix = [1, 2.0, '3']

print(mix)

print(np.array(mix))
only_ones = np.ones((2,3))

print(only_ones)

print(only_ones.dtype.name)
only_ones_of_integer = np.ones((2,3), dtype=np.int16)

print(only_ones_of_integer)

print(only_ones_of_integer.dtype.name)
only_zeros = np.zeros((2,3))

print(only_zeros)
random_arr = np.random.random((2,3))

print(random_arr)
seq = np.arange(5)

print(seq)
np.arange(10,50,5)
np.arange(5,0,-1)
seq = np.linspace(0,10)



print("Number of elements:",seq.size)

print()

print(seq)

start = 5

end = 10

num_of_elements = 10

np.linspace(start, end, num_of_elements)
vector = np.random.random((10,1))

print(vector.shape)

print(vector)
matrix = vector.reshape(2,5)

print(matrix.shape)
print(matrix)
matrix = vector.reshape(2,-1)

print(matrix.shape)
print(matrix)
print(matrix.reshape(10,1))
print(matrix.reshape(-1,1))
flatten_array = matrix.reshape(-1)

print(flatten_array.shape)
print(flatten_array)
arr = np.arange(1,11)

arr
print("Access the first element of an array :",arr[0])

print("Access the fifth element of an arary :",arr[4])

print()

print("Access the last element of an array: ",arr[-1])

print("Access the second element of an array: ",arr[-2])

start_index = 3

end_index = 7



print(arr[ start_index : end_index])
start_index = 0

end_index = 10

step_by = 2

print(arr[ start_index : end_index: step_by])
matrix = np.arange(1,17).reshape(4,4)

print(matrix)
print("Element at first row and first column of matrix:",matrix[0,0])

print("Element at last row and last column of matrix:",matrix[3,3])
print("Second row of the matrix:")

matrix[1]
print("First to third rows of the matrix:")

matrix[0:3]
print("Second column of the matrix:")

matrix[:,1]
print("First to Third columns of the matrix:")

matrix[:,0:3]
print("Find elements of second and third column and second and third row of the matrix")

matrix[1:3, 1:3]
matrix
mask = matrix > 10

mask
matrix[mask]
even_element_mask = matrix % 2 == 0

even_element_mask
matrix[even_element_mask]
matrix = np.arange(1,11).reshape(2,-1)

matrix
transposed_matrix = matrix.T

transposed_matrix
matrix + 10
matrix - 10
matrix * matrix
print("Matrix:",matrix.shape)

print("Transposed matrix: ",transposed_matrix.shape)



matrix.dot(transposed_matrix)
dummy= np.array([[1,2],[3,4]])

dummy
np.linalg.inv(dummy)
h_stacked = np.hstack([matrix, matrix])

h_stacked
v_stacked = np.vstack([matrix, matrix])

v_stacked
arr1, arr2 = np.hsplit(h_stacked, 2)

print("Array1:")

print(arr1)



print("\nArray2:")

print(arr2)
arr1, arr2 = np.vsplit(v_stacked, 2)

print("Array1:")

print(arr1)



print("\nArray2:")

print(arr2)
arr = np.array([[1,1,0,0],[1,1,0,0],[1,1,0,0],[1,1,0,0] ])

print(arr)
np.all(arr==0)
np.all(arr==0, axis=0)
np.all(arr==0, axis=1)
np.any(arr==0)
np.any(arr==0, axis=0)
np.any(arr==0, axis=1)
row_index, col_index = np.nonzero(arr)

print(row_index, col_index)
for r, c in zip(row_index, col_index):

    print("arr[{0}][{1}] = {2}".format(r,c, arr[r][c]))
row_index, col_index = np.nonzero(arr==0)

print(row_index, col_index)
for r, c in zip(row_index, col_index):

    print("arr[{0}][{1}] = {2}".format(r,c, arr[r][c]))
np.where(arr==0)
np.where(arr==0, "zero", "non_zero")
np.where(arr==0, arr, -1)
data =[[11, 0, 3, 4], [34, 5, 1, 9], [-9, 5, 3, 6]]

X = np.array(data)

X
X.max() # maximum element of the array
X.max(axis=0)  # Column-wise maximum elements 
X.max(axis=1) # Row-wise maximum elements
X.min() # miniumn element of the array
X.min(axis=0) # Column-wise minimum elements of the array
X.min(axis=1) # Row-wise minimum elements of the array
X
np.sort(X) # sort array elements in ascending order. By default it's sort the array row-wise
-np.sort(-X) # sort array element in descending order.
np.sort(X, axis=0) # sort elements column-wise
np.sort(X, axis=1) #Sort elements row-wise
X
np.argmax(X) # index of the maximum element of the array
np.argmax(X, axis=0) # indices of the maximum elements of the array column-wise
np.argmax(X, axis=1) # indices of the maximum elements of the array row-wise
X
X.reshape(-1)
np.argmin(X) # returns the index of the minimum element of the array. If axis is not given , it flattens the array first and finds the index. 
np.argmin(X, axis=0) # returns the indices of the minimum elements of the array column-wise
np.argmin(X, axis=1) # returns the indices of the minimum elements of the array row-wise
X
np.argsort(X) # returns the array of indices after sorting the elements in ascending order. 
np.argsort(X, axis=0) # returns the array of indices after sorting the elements in ascending order column-wise
np.argsort(X, axis=1) # returns the array of indices after sorting the elements in ascending order row-wise
X = np.arange(1, 11).reshape(5,2)

X
np.mean(X) # returns mean of the elements on flattened array
np.mean(X, axis=0) # returns the mean of the elements column-wise
np.mean(X, axis=1) # returns the mean of the elements row-wise
np.median(X) # returns median of the elements on flattened array
np.median(X, axis=0) # returns the median of the elements column-wise
np.median(X, axis=1) # returns the median of the elements row-wise
np.var(X) # returns variance of the elements on flattened array
np.var(X, axis=0) # returns the variance of the elements column-wise
np.var(X, axis=1) # returns the variance of the elements row-wise
np.std(X) # returns standard-deviation of the elements on flattened array
np.std(X, axis=0) # returns standard-deviation of the elements column-wise
np.std(X, axis=1) # returns standard-deviation of the elements row-wise