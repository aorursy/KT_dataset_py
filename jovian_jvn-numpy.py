import numpy as np
# Importing the required packages.

import sys

import time

import numpy as np



# NumPy array with 1000 elements

numpy_array = np.arange(1000)



# Python List with 1000 elements (0 to 999)

list = range(0, 1000)



# Size of NumPy array and Python List

print("Size of numpy_array: \n", numpy_array.itemsize * numpy_array.itemsize)

print("Size of list: \n", sys.getsizeof(1)*len(list))
import numpy as np



list = [24, 10, 2019]



# np.array object converts list into NumpPy array

numpy_array1 = np.array(list)

print("List converted into a numpy array : \n", numpy_array1)



# Above operation can be combined without declaring list

numpy_array2 = np.array([17, 11, 1999])

print("A numpy array without declaring list : \n", numpy_array2)



# NumPy array with more than one dimension

numpy_array3 = np.array([[24, 10, 2019], [17, 11, 1999]])

print("A numpy array with more than one dimension : \n", numpy_array3)



# NumPy array with datatype as complex

numpy_array4 = np.array([2, 4, 1], dtype = complex)

print("A numpy array of complex datatype : \n", numpy_array4)
import numpy as np



# A 2x2 NumPy array with all zeros

numpy_array5 = np.zeros((2, 2))

print("A numpy array with all zeros : \n", numpy_array5)



# A 2x2 NumPy array with random values 

numpy_array6 = np.random.random((2, 2))

print("A numpy array with random values : \n", numpy_array6)
import numpy as np



# A NumPy array of the type integer (int32)

numpy_array7 = np.array([24, 10])

print("Data type of the numpy array ",numpy_array7, "\n", numpy_array7.dtype)



# A NumPy array of the type float (float64)

numpy_array8 = np.array([24.10, 10.10])

print("Data type of the numpy array ",numpy_array8, "\n", numpy_array8.dtype)



# A NumPy array of the type complex (complex128)

numpy_array9 = np.array([1 + 2j, 3 + 5j])

print("Data type of the numpy array ",numpy_array9, "\n", numpy_array9.dtype)
import numpy as np



# Single dimension NumPy array of the size, 3

numpy_array10 = np.array([21, 11, 2019])

print("Size/Shape of single dimension numpy array, ", numpy_array10, "\n", numpy_array10.shape)



# 3x3 dimensional NumPy array

numpy_array11 = np.array([[21, 11, 2019], [22, 11, 2019], [23, 11, 2019]])

print("Size/Shape of the 3x3 dimensional numpy array, \n", numpy_array11, "\n", numpy_array11.shape)



# 2x3 dimensional NumPy array

numpy_array12 = np.array([[21, 11, 2019], [22, 11, 2019]])

print(numpy_array12.shape, " dimensional numpy array before resizing \n", numpy_array12)

# Resizing 2x3 dimensional array into a 3x2 dimensional array

numpy_array12.shape = (3, 2)

print("Size/Shape of the numpy array after resizing it \n", numpy_array12.shape)

print("Numpy array after resizing it into 3x2 dimension \n", numpy_array12)
import numpy as np



numpy_array13 = np.array([[10, 11], [12, 13]])

# numpy_array13.ndim returns dimension for the numpy array, numpy_array13

print("A numpy array \n", numpy_array13, "\nwith its dimension as \n", numpy_array13.ndim)
import numpy as np



# Second argument which is dtype which means the items datatype, and it is int8. 

# That means the bytes of each character is 1.

numpy_array14 = np.array([24, 10, 2019], dtype = np.int8)

print(numpy_array14.itemsize)
# Numpy array of data type float32

numpy_array15 = np.array([24, 10, 2019], dtype = np.float32) 

print(numpy_array15.itemsize)
import numpy as np



# Generates a 2x2 dimensional numpy array with random values

numpy_array16 = np.empty([2, 2], dtype = int)

print(numpy_array16)
import numpy as np 



# Generates a 2x2 dimensional array filled with ones

numpy_array17 = np.ones([2,2], dtype = int) 

print(numpy_array17)
import numpy as np



# A numpy array starting from 0 to 9

numpy_array18 = np.arange(10)

print("A numpy array with range 5 \n", numpy_array18)



# A numpy array starting from 1 to 10 (excluding 10) with 2 as spacing between values

numpy_array19 = np.arange(1, 10, 2)

print("A numpy array with start, stop and step parameters \n", numpy_array19)
import numpy as np



# A numpy array of elements from 0 to 9

numpy_array20 = np.arange(10)

print("Numpy array before slicing \n", numpy_array20)

# Slicing elements from 1 to 4 from the numpy array

sliced_index = slice(1,5)

# Numpy array after slicing elements from 1 to 4

print("Numpy array after slicing \n", numpy_array20[sliced_index])
import numpy as np



# Creates 3x4 dimensional numpy array

numpy_array21 = np.array([[24, 25, 26, 27], [28, 29, 30, 31], [32, 33, 34, 35]])

print("Numpy array before slicing \n", numpy_array21)

# Creates a subarray by slicing the first 2 rows and columns 1 and 2

sliced_array = numpy_array21[:2, 1:3]

print("Numpy array after slicing \n", sliced_array)
import numpy as np



# A 3x4 dimensional numpy array

numpy_array22 = np.array([[24, 25, 26, 27], [28, 29, 30, 31], [32, 33, 34, 35]])

print("Numpy array before slicing \n", numpy_array22)

# All column values in the first row

sliced_array1 = numpy_array22[0, :]

print("First row of the array with all column values \n", sliced_array1)

# All column values in the first and second row 

sliced_array2 = numpy_array22[0:2, :]

print("First two rows of the array with all column values \n", sliced_array2)



# Same distinction can be applied when accessing columns of an array

# First column of the array with all row values

sliced_array3 = numpy_array22[:, 0]

print("First column of the array with all row values \n", sliced_array3)

# First two columns of the array with all row values

sliced_array4 = numpy_array22[:, 0:2]

print("First two columns of the array with all row values \n", sliced_array4)
import numpy as np



numpy_array23 = np.array([[24, 25],[26, 27],[28, 29]])

print("Numpy array before integer indexing \n", numpy_array23)

# Selecting elements at (0, 1), (0, 0) and (2, 1) from the numpy_array23.

indexed_array1 = numpy_array23[[0, 0, 2], [1, 0, 1]]

print("Selecting elements at (0, 1), (0, 0) and (2, 1) from the numpy_array23 \n", indexed_array1)



numpy_array24 = np.array([[24, 25, 26], [27, 28, 29], [30, 31, 32], [33, 34, 35]])

print("Numpy array before integer indexing \n", numpy_array24)

# Row indices are (0, 0) and (1, 1)

rows_array = np.array([[0, 0], [1, 1]])

columns_array = np.array([[1, 0], [0, 1]])

indexed_array2 = numpy_array24[rows_array, columns_array]

print("Indexed array \n", indexed_array2)
import numpy as np



numpy_array25 = np.array([[24, 25, 26], [27, 28, 29], [30, 31, 32], [33, 34, 35]])

print("Numpy array before boolean indexing \n", numpy_array25)

print("Numpy array with elements grater than 30 \n", numpy_array25[numpy_array25 > 30])
import numpy as np



# Creates a 3x3 dimensional numpy array with numbers from 0 to 8

numpy_array26 = np.arange(9).reshape(3, 3)

print("First numpy array \n", numpy_array26)



# Creates a single dimension numpy array

numpy_array27 = np.array([5, 5, 5])

print("Second numpy array \n", numpy_array27)



print("Adding two numpy arrays \n", np.add(numpy_array26, numpy_array27))

print("Subtracting two numpy arrays \n", np.subtract(numpy_array26, numpy_array27))

print("Multiplying two numpy arrays \n", np.multiply(numpy_array26, numpy_array27))

print("Dividing two numpy arrays \n", np.divide(numpy_array26, numpy_array27))
import numpy as np



numpy_array28 = np.array([10, 20, 30])

print("First numpy array \n", numpy_array28)

numpy_array29 = np.array([4, 5, 6])

print("Second numpy array \n", numpy_array29)

numpy_array30 = np.array([24.8, 21.1, 90.4])

print("Third numpy array \n", numpy_array30)



print("Modulus after dividing the arrays \n", np.mod(numpy_array28, numpy_array29))

print("Applying power function on the first numpy array \n", np.power(numpy_array28, 2))

print("Applying around function on the third numpy array \n", np.around(numpy_array30))

print("Applying floor function on the third numpy array \n", np.floor(numpy_array30))

print("Applying ceil function on the third numpy array \n", np.ceil(numpy_array30))
import numpy as np



numpy_array31 = np.arange(9)

print("First numpy array \n", numpy_array31)



print("Applying average function on the first numpy array \n", np.average(numpy_array31))

print("Applying mean function on the first numpy array \n", np.mean(numpy_array31))

print("Applying median function on the first numpy array \n", np.median(numpy_array31))

print("Applying minimum function on the first numpy array \n", np.amin(numpy_array31))

print("Applying maximum function on the first numpy array \n", np.amax(numpy_array31))

print("Applying standard deviation function on the first numpy array \n", np.std(numpy_array31))

print("Applying variance function on the first numpy array \n", np.var(numpy_array31))
import numpy as np



numpy_array32 = np.array([24, 25])

numpy_array33 = np.array([26, 27])



# (24 * 26) + (25 * 27)

print("Dot product of the arrays \n", np.dot(numpy_array32, numpy_array33))



numpy_array34 = np.array([[24, 25], [26, 27]])

numpy_array35 = np.array([[28, 29], [30, 31]])

print("Matrix Multiplication of the arrays \n", np.matmul(numpy_array34, numpy_array35))



print("Determinant of the 2x2 dimensional array [[28, 29], [30, 31]] \n", np.linalg.det(numpy_array35))
import numpy as np



# One dimensional numpy array

numpy_array36 = np.array([24, 25, 26])

print("One dimensional numpy array \n", numpy_array36)

value = 3

print("Scalar value \n", value)

print("Resulted array after adding a scalar value and one dimensional numpy array \n", (numpy_array36 + value)) 



# Two dimensional numpy array

numpy_array37 = np.array([[24, 25, 26], [27, 28, 29]])

print("Two dimensional numpy array \n", numpy_array37)

print("Resulted array after adding a scalar value and two dimensional numpy array \n", (numpy_array37 + value))

print("Resulted array after adding one dimensional numpy array and two dimensional numpy array \n", (numpy_array36 + numpy_array37))
from scipy.misc import imread, imsave, imresize

from IPython.display import Image



# Reading a JPEG image into a numpy array

img = imread('2020.jpg')

print("Datatype of the image \n", img.dtype)

print("Shape of the image \n", img.shape)



# We can tint the image by scaling each of the color channels

# by a different scalar constant. The image has shape (400, 248, 3);

# we multiply it by the array [1, 0.95, 0.9] of shape (3,);

# numpy broadcasting means that this leaves the red channel unchanged,

# and multiplies the green and blue channels by 0.95 and 0.9

# respectively.

img_tinted = img * [1, 2, 3]



# Resize the tinted image to be 300 by 300 pixels.

img_tinted = imresize(img_tinted, (300, 300))



# Write the tinted image back to disk

imsave('2020_modified.jpg', img_tinted)