#Basic array creation

import numpy as np

#numpy.array(object, dtype=None, copy=True, order='K',.....)

a = np.array([2,3,4])

print(a)

# Checking the data type

print(a.dtype)

b = np.array([1.2, 3.5, 5.1])

print(b.dtype)
b = np.array([[1.5,2,3], [4,5,6]])

print(b)

# finding size of an array

print(b.size)
#Sequence of Numbers

print(np.arange( 10, 30, 5 ))

print(np.arange( 0, 2, 0.3 ))

print(np.arange( 5 ))

# Question: What are the default values?
# Random Number

np.random.rand(1)  # Creates a randome number between 0 to 1 and assumes uniform distribution

np.random.rand(5,3) # Creating a Matrix

np.random.randn(5,5) # Creating numbers from a normal distribution

np.random.randint(1,100,10)

np.random.seed(42)
#array with string

string_arr=np.array(['Ganga','Jamuna','Kaveri'])

string_arr[1]='Godavari'

string_arr

# Assignment change the second element to Godavari and print the array
print(a)

#Accesing Single Element

print(a[2])



#Accesing Multiple Elements

# First and second element

print(a[0:2]) 

# Any one bound can be used

print(a[: 3])

print(a[1: ])

# If the end limit crosses the bound of the array

print(a[0:200])

# Assigning to another array

b=a[0:5]

# What happens if we do not specify neither start and stop

b[:]=99



# Element wise operation

a = np.arange(10)**3

a= a*2

print(a)
l=[1,2,3]

print(l*2)

a=np.array([1,2,3])

print(a*2)
a = np.array([2,3,4,25])

b=np.arange( 0, a.size, 2)

b

print(a[b])
# 2D Array

a2d=np.array([[1,3,5],[2,5,8],[3,6,9]])

# Accessing the element 8

a2d[1][2]

a2d[1,2]

# Aceessing 0th and 1st row and 1st and 2nd columns

a2d[:2,1:]
arrlog= np.arange(2,25,2)

# Creating boolean indicator

arrlog>10

# Accesing elements based on 

arrlog[arrlog>10]
a2d.sum()

# Can we do row wise or column wise sum

a2d.sum(axis = 0)

a2d.sum(axis = 1)
print (np.zeros( (3,4) ))

print (np.ones( (2,3,4), dtype=np.int16 ))

print (np.empty( (2,3)))
b = np.arange(12).reshape(4,3)

print(b)

c = np.arange(24).reshape(2,3,4)

print(c)
# Two sepecial cases are NaN and Inf

special=np.array([-1,0,1])

special=special/0

print(special)

#How to detect if such cases are presnet already in a big Array

# Equalto does not work with NaN

np.isnan(special)

np.isinf(special)
#Basic Operations

a = np.array( [20,30,40,51] )

b = np.arange( 4 )

print (b)

c = a-b

print (c)

#Operations on all elements

print(b**2)

print(10*np.sin(a))

print(a[a<35])

# Assigment Declare an NP Array, Find Sum all odd elements

#Matrix Operations

A = np.array( [[1,1],

            [0,1]] )

B = np.array( [[2,0],

               [3,4]] )



# Elementwise multiplication

print(A*B)

# Matrix multiplication

print(A @ B )

print(A.dot(B))
#Finding the array type

a = np.ones(3, dtype=np.int32)

# Returns num evenly spaced samples, calculated over the interval [start, stop].

# .linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0)[source]

b = np.linspace(0,3.14,3)

print(b.dtype.name)

c = a+b

print(c)

print(c.dtype.name)

d = np.exp(c*1j)

print (d)

print(d.dtype.name)
#Unary operations

a = np.random.random((2,3))

print(a)

print(a.sum())

print(a.min())

print(a.max())
#Universal functions

B = np.arange(3)

print(B)

print(np.exp(B))

print(np.sqrt(B))

C = np.array([2., -1., 4.])

print(C)
#Shape Manipulation

a = np.floor(10*np.random.random((3,4)))

print(a)

print(a.shape)

# numpy.ravel(a, order='C')[source]¶

# Return a contiguous flattened array.

print(a.ravel())

print(a.reshape(6,2))

# Same as self.transpose(), except that self is returned if self.ndim < 2.

print(a.T)

print(a.T.shape)

print(a.shape)
#Stacking together different arrays 

a = np.floor(10*np.random.random((2,2)))

print(a)

b = np.floor(10*np.random.random((2,2)))

print(b)

# Stack arrays in sequence vertically (row wise).

print(np.vstack((a,b)))

# Stack arrays in sequence horizontally (column wise).

print(np.hstack((a,b)))
#Splitting one array into several smaller ones

x = np.arange(16.0).reshape(4, 4)

print(np.hsplit(x,2))
#Copies and Views

a = np.arange(12)

b=a

print(b is a)

b.shape = 3,4 

print(a.shape)

# New view of array with the same data.

c = a.view()

print(c is a)

print(c.base is a)

print(c.flags.owndata)

c.shape = 2,6

print(a.shape)

c[0,4] = 1234

print(a)

d = a.copy() 

# Return a copy of the array.

print(d is a)

print(d.base is a)

d[0,0] = 9999

print(a)
# Identity matrix

a= np.eye(5)