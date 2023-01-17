# Load NumPy Library 
import numpy as np 
# Create a vector as row 
vector_row = np.array([1, 2, 3])
print(vector_row) 
# Create a vector as column 
vector_column = np.array([[1], [2], [3]]) 
print(vector_column) 
mat1 = np.matrix("1, 2, 3, 4; 4, 5, 6, 7; 7, 8, 9, 10")
print(mat1)
mat2 = np.array([[1, 2], [3,4], [4, 6]])
print(mat2) 
mat3 = np.matrix("1, 2, 3, 4; 4, 5, 6, 7; 7, 8, 9, 10")
# shape 
mat3.shape
# rows 
mat3.shape[0]
# columns 
mat3.shape[1]
mat4 = np.matrix("1, 2, 3, 4; 4, 5, 6, 7; 7, 8, 9, 10")
# size 
mat4.size
mat5 = np.matrix("1, 2, 3, 4; 4, 5, 6, 7; 7, 8, 9, 10")
print(mat5)
# adding a new matrix `col_new` as a new column to mat5
col_new = np.matrix("1, 1, 1")
print(col_new)
# insert at column 
mat6 = np.insert(mat5, 0, col_new, axis=1)
print(mat6) 
# adding a new matrix `row_new` as a new row to mat5
row_new = np.matrix("0, 0, 0, 0")
print(row_new)
# insert at row 
mat7 = np.insert(mat5, 0, row_new, axis=0)
print(mat7)
mat_a = np.matrix("1, 2, 3, 4, 5; 5, 6, 7, 8, 9; 9, 10, 11, 12, 13")
print(mat_a)
# change 6 with 0 
mat_a[1, 1] = 0 
# show mat_a 
print(mat_a)
# extract 2nd row 
mat_a[1, :]
# extract 3rd column
mat_a[:, 2]
# extract elements 
mat_a[1, 2]
A = np.arange(0, 20).reshape(5,4)
print(A)
B = np.arange(20, 40).reshape(5,4)
print(B)
# addition 
np.add(A, B)
# subtraction 
np.subtract(A,B)
A = np.arange(0, 20).reshape(5,4)
print(A)
# transpose 
np.transpose(A)
# multiplication
np.dot(A,B) 
# transpose matrix B to make it 4x5 in dimension
T = np.transpose(B)
print(T)
# now we can perform multiplication
np.dot(A,T)
# using matmul 
np.matmul(A, T)
# using @ operator 
A @ T 
# element-wise multiplication 
np.multiply(A, B)
# division
np.divide(A, B)