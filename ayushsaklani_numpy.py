# Convention for import to get shortened namespace

import numpy as np
# Create a simple array from a list of integers

a = np.array([1, 2, 3])

a
# See how many dimensions the array has

a.ndim
# Print out the shape attribute

a.shape
# Print out the data type attribute

a.dtype
# This time use a nested list of floats

a = np.array([[1., 2., 3., 4., 5.]])

a
# See how many dimensions the array has

a.ndim
# Print out the shape attribute

a.shape
# Print out the data type attribute

a.dtype
a = np.arange(5)

print(a)
a = np.arange(3, 11)

print(a)
a = np.arange(1, 10, 2)

print(a)
b = np.linspace(5, 15, 5)

print(b)
b = np.linspace(2.5, 10.25, 11)

print(b)
a = range(5, 10)

b = [3 + i * 1.5/4 for i in range(5)]

b
a = []

for i in range(5):

    a.append(i)

a

    
result = []

for x, y in zip(a, b):

    result.append(x + y)

print(result)
a = np.arange(5, 10)

b = np.linspace(3, 4.5, 5)
a + b
a * b
a = np.array([1,2,3,4,5])

a=a*2

print(a)

np.pi
np.e
# This makes working with radians effortless!

t = np.arange(0, 2 * np.pi + np.pi / 4, np.pi / 4)

t
# Calculate the sine function

sin_t = np.sin(t)

print(sin_t)
# Round to three decimal places

print(np.round(sin_t, 3))
# Calculate the cosine function

cos_t = np.cos(t)

print(cos_t)
# Convert radians to degrees

degrees = np.rad2deg(t)

print(degrees)
# Integrate the sine function with the trapezoidal rule

sine_integral = np.trapz(sin_t, t)

print(np.round(sine_integral, 3))
# Sum the values of the cosine

cos_sum = np.sum(cos_t)

print(cos_sum)
# Calculate the cumulative sum of the cosine

cos_csum = np.cumsum(cos_t)

print(cos_csum)
# Convention for import to get shortened namespace

import numpy as np
# Create an array for testing

a = np.arange(12).reshape(3, 4)
a
a[1, 2]
a[2]
a[0, -1]
# Get the 2nd and 3rd rows

a[1:3]
# All rows and 3rd column

a[:, 2]
# ... can be used to replace one or more full slices

a[..., 2]
# Slice every other row

a[::2]
a
a*3