import numpy as np
# 1D Array
a = np.array([0, 1, 2, 3, 4])
b = np.array((0, 1, 2, 3, 4))
c = np.arange(5)
d = np.linspace(0, 2*np.pi, 5)
print(a) 
print(b) 
print(c) 
print(d) 
print(a[3]) 
# MD Array,
a = np.array([[11, 12, 13, 14, 15],
              [16, 17, 18, 19, 20],
              [21, 22, 23, 24, 25],
              [26, 27, 28 ,29, 30],
              [31, 32, 33, 34, 35]])

print(a[2,4]) 
# MD slicing
print(a[0, 1:4]) 
print(a[1:4, 0]) 
print(a[:, 1])
# Array properties
a = np.array([[11, 12, 13, 14, 15],
              [16, 17, 18, 19, 20],
              [21, 22, 23, 24, 25],
              [26, 27, 28 ,29, 30],
              [31, 32, 33, 34, 35]])

print(type(a)) 
print(a.dtype) 
print(a.size) #number of Elements 
print(a.shape) # rows x columns
print(a.ndim) # number of dimensions
# Basic Operators
a = np.arange(25)
a = a.reshape((5, 5))

b = np.array([10, 62, 1, 14, 2, 56, 79, 2, 1, 45,
              4, 92, 5, 55, 63, 43, 35, 6, 53, 24,
              56, 3, 56, 44, 78])
b = b.reshape((5,5))

print('Addition')
print(a + b)
print('Subtraction')
print(a - b)
print('Multiplication')
print(a * b)
print('Division')
print(a / b)
print('Power')
print(a ** 2)
print('Less than')
print(a < b)
print('Greater than')
print(a > b)
print('Dot Product')
print(a.dot(b))
# dot, sum, min, max, cumsum
a = np.arange(10)

print(a.sum()) 
print(a.min()) 
print(a.max()) 
print(a.cumsum()) # Elementwise addition
