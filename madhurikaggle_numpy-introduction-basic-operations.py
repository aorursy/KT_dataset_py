#import numpy library.

import numpy as np
#using list.

data = [1, 2, 3, 4, 5]

#using np.array function.

arr = np.array(data)
arr
data1 = [[1, 2, 3, 4],[5, 6, 7, 8]]

arr1 = np.array(data1)
arr1
#shape of an array.

print("arr1:")

print(arr1)

print("The shape of arr1 is ", arr1.shape)
type(arr1)
arr1.dtype
# to get dimension of an array.

arr1.ndim
arr1.size #return size of arr1. size = rows x columns
arr1.itemsize #return a size of each elements in byte.
arr1.data
#creating array using tuple.

arr2 = np.array(((1, 2, 3), (4, 5, 6)))

arr2
arr = np.array([(1, 2, 3), [4, 5, 6], (7, 8, 9)])

arr
str_data = np.array([['a', 'b'], ['c','d']])

str_data
#create an array with complex values using dtype option.

np.array([[1, 2, 3],[4, 5, 6]], dtype=complex)
#zero() function.

np.zeros((3,3)) #shape=(3,3) row=3, columns = 3.
#ones

np.ones((3,3))
np.arange(0,10)
np.arange(10)#starting value defaulted to 0.
#with two arguments

np.arange(3,10)
np.arange(0, 11, 2) #array of even numbers.
# this third argument can be 'float' type.

np.arange(1, 5, 0.5)
np.arange(0, 12).reshape(3,4)
np.linspace(0, 10, 5)
#one-dimensional array.

np.random.random(3)
#multi-dimensional array.

np.random.random((3,4))
a = np.arange(10, 15)

a
#addition with scalar

a + 2
a * 2 #multiplication
a - 2 #subtraction
a / 2 #division.
a = np.arange(1, 5)

b = np.arange(5, 9)

print('a =', a)

print('b =',b)

print('a + b = ', a + b) #element-wise addition
print('b =',b)

print('a =', a)

print('b - a =', b - a) #elemenet-wise subtraction
print(a * b) #element-wise multiplication
print('a =', a)

print('b =', b)
a * np.sin(b)
a * np.sqrt(b)
A = np.arange(1, 10).reshape(3, 3)

A
B = np.ones((3, 3))

B
#element-wise multiplication on Multidimensional arrays

# A and B.

A * B
np.dot(A, B)
#using dot() function.

A.dot(B)
# A * B not equal to B * A



np.dot(B, A) # similar to B.dot(A)
a = np.arange(5)

a
a += 1

a
a -= 1

a
a = np.array([1, 2, 3, 4])

print('a =', a)
np.sqrt(a)
np.log(a)
np.sin(a)
a = np.array([1, 2, 3, 4])

print('a =', a)
a.sum() #return sum of all elements in array a.
a.min() #return smallest value form array a.
a.max() #return largest value from array a.
a.mean() #return avarage of array a.
a.std() #return standard deviation of array a.