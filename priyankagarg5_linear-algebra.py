# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# create a vector
a = np.array([1, 2, 3])
print(a)


#multiply a vector
x=np.array([1,2,3])
y=np.array([1,2,3])
mulv=x*y
print(mulv)
#addition of vectors
addv=x+y
print(addv)

#divison of vectors
divv=x/y
print(divv)

#subtraction of vectors
subv=x-y
print(subv)
#vector dot product
dot_prodv=np.dot(x,y)
print(dot_prodv)
#create matrix
m = np.array([[1,2,3],[4,5,6]])
print(m)
#add matrices
a=np.array([[1,2,3],[4,5,6]])
b=np.array([[1,2,3],[4,5,6]])
add=a+b
print(add)
#matrix dot product
m1=np.array([[1,2],[3,4],[5,6]])
m2=np.array([[1,2,3],[4,5,6]])
dot_prod_matrix= m1.dot(m2)
print(dot_prod_matrix)
#subtraction of matrices
sub=a-b
print(sub)
#divison of matrices
div=np.divide(a,b)
print(div)

#hadamard product
#element-wise multiplication
mul=a*b
print(mul)
#vector-matrix multiplication
vec_mat_mul=np.dot(a,x)
print(vec_mat_mul)
#transpose of matrix
trans=a.T
print(trans)
#inverse of matrix
matrix=np.array([[1,2],[3,4]])
inverse=np.linalg.inv(matrix)
print(inverse)
#determinant of matrix
det=np.linalg.det(matrix)
print(det)
#trace
trace=np.trace(matrix)
print(trace)
#rank
rank=np.linalg.matrix_rank(matrix)
print(rank)

