#import library numpy.

import numpy as np
#create an array a.

a = np.arange(10,15)

a
#for one-dimensional array.

#accessing elements using indexing.

a[2]  #select element at index 2.
a[-1] #return a last element of an array a.
a[-3]
a[[2, 1, 4]] #advanced indexing.
#creating multi-dimensional array.

A = np.arange(1,10).reshape((3, 3))

A
#to access the element 7 from matrix A.

A[2, 0] # third row, and first column.
#for one dimensional array.

a
a[1:4]
a[0:4:2]
a[:4:2] #without providing starting index which defaulted as 0.
#accessing all elements with step 2.

a[::2]
a[:] #accessing all elements.
A
#to access the first row.

A[0, :]
#to access the first column.

A[:, 0]
#to acess smaller matrix.

A[1:3, 0:2]
a
#for one-dimensional array.

for i in a:

    print(i)
#Iterating through matrix.

A
for i in A:

    print(i)
for i in A:

    for j in i:

        print(j)
for item in A.flat:

    print(item)
A
#calculating average of the values by columns.

np.apply_along_axis(np.mean, axis=0, arr=A)
#calculating average of values by rows.

np.apply_along_axis(np.mean, axis = 1, arr = A)
def foo(x):

    return x/2



np.apply_along_axis(foo, axis=0, arr=A)
np.apply_along_axis(foo, axis=1, arr=A)
# creating ramdom 4x4 matrix using 

# ramdom of np.random module of numpy.



B = np.random.random((4,4))

B
B < 0.5
B[B < 0.5]
a = np.random.random(10)

a
A = a.reshape(5, 2)

A
a.shape = (5, 2)

a
a
a.ravel()
a.shape = (10)

a
A
A_transpose = A.transpose()

A_transpose
#let's check the shape.

print("Shape of A before transposing:",A.shape)

print("Shape of A after transposing:",A_transpose.shape)