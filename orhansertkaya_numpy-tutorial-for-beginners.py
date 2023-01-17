import numpy as np
np.array([3.2,4,6,5])
np.array([1,4,2,5,3]) ## integer array:
np.array([1,2,3,4], dtype="str")
np.array([3,6,2,3], dtype="float32")
# nested lists result in multidimensional arrays



np.array([range(i,i+3) for i in [2,4,6]])
# Create a length-10 integer array filled with zeros



np.zeros(10, dtype="int") 
np.zeros((5,6), dtype="float")
# Create a 3x5 floating-point array filled with 1s



np.ones((3,5), dtype="float")
# Create a 3x5 array filled with 3.14



np.full((3,5), 3.14)
# Create an array filled with a linear sequence

# Starting at 0, ending at 20, stepping by 2

# (this is similar to the built-in range() function)



np.arange(0,20,2)
# Create an array of five values evenly spaced between 0 and 1



np.linspace(0,1,5)
# Create a 3x3 array of uniformly distributed

# random values between 0 and 1



np.random.random((3,3))
# Create a 3x3 array of normally distributed random values

# with mean 0 and standard deviation 1



np.random.normal(0,1,(3,3))
# Create a 3x3 array of random integers in the interval [0, 10)



np.random.randint(0,10,(3,3))
# Create a 3x3 identity matrix



np.eye(3)
#Return a new array of given shape and type, with random values



np.empty((3,3),dtype="int")
np.zeros(10,dtype="int16")
#or using the associated NumPy object:



np.zeros(10,dtype=np.int16)
#NumPy Array Attributes

#We’ll use NumPy’s random number generator, which we will seed with a set value in order to ensure that the same random arrays are generated each time this code is run:



np.random.seed(0) # seed for reproducibility

x1 = np.random.randint(10, size=6) # One-dimensional array
#Each array has attributes ndim (the number of dimensions), shape (the size of each dimension), and size (the total size of the array):



np.random.seed(0) # seed for reproducibility

x1 = np.random.randint(10, size=6) #it's same ((np.random.randint((0,10), size=6))) # One-dimensional array

x2 = np.random.randint(10, size=(3,4)) # Two-dimensional array

x3 = np.random.randint(10, size=(3,4,5)) # Three-dimensional array



print("x1 ndim: ",x1.ndim)

print("x1 shape: ",x1.shape)

print("x1 size: ",x1.size) #totaly,6 elements



print("x1 ndim: ",x2.ndim)

print("x1 shape: ",x2.shape)

print("x1 size: ",x2.size) #totaly,12 elements



print("x1 ndim: ",x3.ndim)

print("x1 shape: ",x3.shape)

print("x1 size: ",x3.size)#totaly,60 elements



print("dtype: ",x1.dtype) #the data type of the array

# Other attributes include itemsize, which lists the size (in bytes) of each array element,

# and nbytes, which lists the total size (in bytes) of the array:

print("itemsize:",x1.itemsize,"bytes")

print("nbytes:",x1.nbytes,"bytes")



print("dtype: ",x2.dtype) #the data type of the array

print("itemsize:",x2.itemsize,"bytes")

print("nbytes:",x2.nbytes,"bytes")



print("dtype: ",x3.dtype) #the data type of the array

print("itemsize:",x3.itemsize,"bytes")

print("nbytes:",x3.nbytes,"bytes") 



#In general, we expect that nbytes is equal to itemsize times size.
x1
x1[0]
x1[4]
#To index from the end of the array, you can use negative indices:



x1[-1]
x1[-2]
x2
x2[2,1]
x2[2,0]
x2[2,-4]
x2[-2,-3]
#You can also modify values using any of the above index notation:



x2[0,0]=12

x2
x1
x1[0] = 3.14159 # this will be truncated!

x1
x = np.arange(10)

x
x[:5] # first five elements
x[5:] # elements after index 5
x[4:7]# middle subarray
x[::2] # every other element
x[1::2] #every other element, starting at index 1
x[-7:-2:2]
x[-4:-2:1]
# A potentially confusing case is when the step value is negative. In this case, the

# defaults for start and stop are swapped. This becomes a convenient way to reverse

# an array:



x[::-1] # all elements, reversed
x[5::-2]# reversed every other from index 5
x[5:1:-2]
x[5:-8:-1]
x[7:-6:-1]
x[-7:-8:-1]
# Multidimensional slices work in the same way, with multiple slices separated by commas.

# For example:



x2
# two rows, three columns



x2[:2, :3]
# all rows, every other column



x2[:3,::2]
#Finally, subarray dimensions can even be reversed together:



x2[::-1,::-1]
# One commonly needed routine is accessing single

# rows or columns of an array. You can do this by combining indexing and slicing,

# using an empty slice marked by a single colon (:):



print(x2[:, 0]) # first column of x2
print(x2[0,:]) # first row of x2
#In the case of row access, the empty slice can be omitted for a more compact syntax:



print(x2[0]) # equivalent to x2[0, :]
print(x2)
#Let’s extract a 2×2 subarray from this:



x2_sub = x2[:2,:2]

print(x2_sub)
#Now if we modify this subarray, we’ll see that the original array is changed! Observe:



x2_sub[0,0] = 99

print(x2_sub)
print(x2)
x2_sub_copy = x2[:2,:2].copy()

print(x2_sub_copy)
#If we now modify this subarray, the original array is not touched:



x2_sub_copy[0,0] = 42

print(x2_sub_copy)
print(x2)
# Another useful type of operation is reshaping of arrays. The most flexible way of

# doing this is with the reshape() method. For example, if you want to put the numbers

# 1 through 9 in a 3×3 grid, you can do the following:



grid = np.arange(1,10,1).reshape(3,3)

print(grid)
x = np.array([1, 2, 3])

x.shape # x is a vector (3,)
# row vector via reshape



x.reshape(1,3).shape
# row vector via newaxis



x[np.newaxis, :].shape
x.reshape(1,-1).shape
# column vector via reshape



x.reshape((3, 1))
# column vector via newaxis



x[:, np.newaxis]
x.reshape(-1,1).shape
x = np.array([1,2,3])

y = np.array([3,2,1])

np.concatenate((x, y))

z = np.array([99,99,99]) #z =[99,99,99]



print(np.concatenate((x,y,z)))
grid = np.array([[1,2,3],

                [4,5,6]])
# concatenate along the first axis



np.concatenate((grid,grid))
# concatenate along the second axis (zero-indexed)



np.concatenate((grid, grid), axis=1)
# For working with arrays of mixed dimensions, it can be clearer to use the np.vstack

# (vertical stack) and np.hstack (horizontal stack) functions:



x = np.array([1,2,3])

grid = np.array([[9,8,7],

                 [6,5,4]])





# vertically stack the arrays

np.vstack([x,grid])
#horizontally stack the arrays



y = np.array([[99],

            [99]])

np.hstack([grid,y])
x = [1,2,3,99,99,3,2,1]

x1, x2, x3 = np.split(x, [3,5])

print(x1, x2, x3)
x = np.array([1,2,3,99,99,3,2,1])

x1, x2, x3, x4 = np.split(x, [3,4,5])

print(x1, x2, x3,x4)
grid = np.arange(36,dtype=np.float).reshape((6,6))

grid
upper, lower = np.vsplit(grid, [2])

print(upper)

print(lower)
upper,middle, lower = np.vsplit(grid, [2,3])

print("upper: ",upper)

print("middle: ",middle)

print("lower: ",lower)
left, right = np.hsplit(grid, [2])

print(left)

print(right)
left, right = np.hsplit(grid, 2)

print(left)

print(right)
# NumPy’s ufuncs feel very natural to use because they make use of Python’s native

# arithmetic operators. The standard addition, subtraction, multiplication, and division

# can all be used:



x = np.arange(4)

print("x =", x)

print("x + 5 =", x + 5)

print("x - 5 =", x - 5)

print("x * 2 =", x * 2)

print("x / 2 =", x / 2)

print("x // 2 =", x // 2) # floor division
#There is also a unary ufunc for negation, a ** operator for exponentiation, and a %

#operator for modulus:



print("-x = ", -x)

print("x ** 2 = ", x ** 2)

print("x % 2 = ", x % 2)
# In addition, these can be strung together however you wish, and the standard order

# of operations is respected:



-(0.5*x+1) ** 2
# All of these arithmetic operations are simply convenient wrappers around specific

# functions built into NumPy; for example, the + operator is a wrapper for the add

# function:



print(np.add(3,2))



print(np.add(x,2)) #Addition +

print(np.subtract(x,5)) #Subtraction -

print(np.negative(x)) #Unary negation -

print(np.multiply(x,3)) #Multiplication *

print(np.divide(x,2)) #Division /

print(np.floor_divide(x,2)) #Floor division //

print(np.power(x,2)) #Exponentiation **

print(np.mod(x,2)) #Modulus/remainder **



print(np.multiply(x, x))
# Just as NumPy understands Python’s built-in arithmetic operators, it also understands

# Python’s built-in absolute value function:



x = np.array([-2,-1,0,1,2])

abs(x)
# The corresponding NumPy ufunc is np.absolute, which is also available under the

# alias np.abs:



print(np.absolute(x))

print(np.abs(x))
# This ufunc can also handle complex data, in which the absolute value returns the

# magnitude:



x = np.array([7-24j,4-3j,2+0j,1+3j])

np.abs(x)
# NumPy provides a large number of useful ufuncs, and some of the most useful for the

# data scientist are the trigonometric functions. We’ll start by defining an array of

# angles:



theta = np.linspace(0,np.pi,3)





#Now we can compute some trigonometric fuctions on these values:

print("theta      =",theta)

print("sin(theta) =",np.sin(theta))

print("cos(theta) =",np.cos(theta))

print("tan(theta) =",np.tan(theta))
x = [-1, 0, 1]



print("x = ", x)

print("arcsin(x) = ", np.arcsin(x))

print("arccos(x) = ", np.arccos(x))

print("arctan(x) = ", np.arctan(x))
x = [1,2,3]

print("x      =",x)

print("e^x    =",np.exp(x))

print("2^x    =",np.exp2(x))

print("3^x    =",np.power(3,x))
# The inverse of the exponentials, the logarithms, are also available. The basic np.log

# gives the natural logarithm; if you prefer to compute the base-2 logarithm or the

# base-10 logarithm, these are available as well:



x = [1, 2, 4, 10]

print("x        =", x)

print("ln(x)    =", np.log(x))

print("log2(x)  =", np.log2(x))

print("log10(x) =", np.log10(x))
# There are also some specialized versions that are useful for maintaining precision

# with very small input:



x = [0, 0.001, 0.01, 0.1]

print("exp(x) - 1 =", np.expm1(x))

print("log(1 + x) =", np.log1p(x))
x = np.arange(5)

y = np.empty(5)

np.multiply(x, 10, out=y)

print(y)
#This can even be used with array views. For example, we can write the results of a

#computation to every other element of a specified array:



y = np.zeros(10)

np.power(2, x, out=y[::2])

print(y)



# If we had instead written y[::2] = 2 ** x, this would have resulted in the creation

# of a temporary array to hold the results of 2 ** x, followed by a second operation

# copying those values into the y array. This doesn’t make much of a difference for such

# a small computation, but for very large arrays the memory savings from careful use of

# the out argument can be significant.
y = np.zeros(10)

y[::2] = 2 ** x

print(y)
x = np.arange(1,6)

print(np.add.reduce(x))

print(np.subtract.reduce(x))

print(np.multiply.reduce(x))
#If we’d like to store all the intermediate results of the computation, we can instead use

#accumulate:



print(np.add.accumulate(x))

print(np.subtract.accumulate(x))

print(np.multiply.accumulate(x))

print(np.divide.accumulate(x))

print(np.floor_divide.accumulate(x))
x = np.arange(1,6)

np.multiply.outer(x, x)
# As a quick example, consider computing the sum of all values in an array. Python

# itself can do this using the built-in sum function:



L = np.random.random(100)

sum(L)
#The syntax is quite similar to that of NumPy’s sum function, and the result is the same

#in the simplest case:



np.sum(L)
# However, because it executes the operation in compiled code, NumPy’s version of the

# operation is computed much more quickly:



big_array = np.random.rand(1000000)

%timeit sum(big_array)

%timeit np.sum(big_array)
#Similarly, Python has built-in min and max functions, used to find the minimum value

#and maximum value of any given array:



min(big_array),max(big_array)
#NumPy’s corresponding functions have similar syntax, and again operate much more

#quickly:



np.min(big_array),np.max(big_array)
%timeit min(big_array)

%timeit np.min(big_array)
# For min, max, sum, and several other NumPy aggregates, a shorter syntax is to use

# methods of the array object itself:



print(big_array.min(), big_array.max(), big_array.sum())

# Whenever possible, make sure that you are using the NumPy version of these aggre‐

#gates when operating on NumPy arrays!

%timeit np.min(big_array)

%timeit big_array.min()
# One common type of aggregation operation is an aggregate along a row or column.

# Say you have some data stored in a two-dimensional array:



M = np.random.random((3,4))

print(M)



M.sum()
# Aggregation functions take an additional argument specifying the axis along which

# the aggregate is computed. For example, we can find the minimum value within each

# column by specifying axis=0:



print(M.min(axis=0))

#or use that way

print(np.min(M,axis=0))

M
# Similarly, we can find the maximum value within each row:



M.max(axis=1)
# Note that some of these NaN-safe functions were not added until

# NumPy 1.8, so they will not be available in older NumPy versions.



x = np.array([1,2,np.nan,4,5])



print("np.sum       =",np.sum(x))

print("np.nansum    =",np.nansum(x))



print("np.mean      =",np.mean(x))

print("np.nanmean   =",np.nanmean(x))



print("np.std       =",np.std(x))

print("np.nanstd    =",np.nanstd(x))





#Be careful that this is not a real index of minimum value.

print("np.argmin    =",np.argmin(x)) 

#if there is a nan value in an array, it returns index of nan value.





print("np.nanargmin =",np.nanargmin(x))

import numpy as np



a = np.array([0,1,2])

b = np.array([5,5,5])

a+b
a+5
# We can similarly extend this to arrays of higher dimension. Observe the result when

# we add a one-dimensional array to a two-dimensional array:



M = np.ones((3,3))

M
M+a



# Here the one-dimensional array a is stretched, or broadcast, across the second

# dimension in order to match the shape of M .
# here we’ve stretched both a and b to match a common shape, and the result is a two-

# dimensional array!



a = np.arange(3) #(3,) 1 dimensional

b = np.arange(3)[:,np.newaxis] #(3,1) 2 dimensional

print(a)

print(b)
a+b
#Let’s look at adding a two-dimensional array to a one-dimensional array:

M = np.ones((2,3))

a = np.arange(3)



# Let’s consider an operation on these two arrays. The shapes of the arrays are:

# M.shape = (2, 3)

# a.shape = (3,)

# We see by rule 1 that the array a has fewer dimensions, so we pad it on the left with

# ones:

# M.shape -> (2, 3)

# a.shape -> (1, 3)

# By rule 2, we now see that the first dimension disagrees, so we stretch this dimension

# to match:

# M.shape -> (2, 3)

# a.shape -> (2, 3)

# The shapes match, and we see that the final shape will be (2, 3) :



M+a
# Let’s take a look at an example where both arrays need to be broadcast:

a = np.arange(3).reshape((3,1))

b = np.arange(3)

# Again, we’ll start by writing out the shape of the arrays:



# a.shape = (3, 1)

# b.shape = (3,)

# |

# Rule 1 says we must pad the shape of b with ones:

# a.shape -> (3, 1)

# b.shape -> (1, 3)

# And rule 2 tells us that we upgrade each of these ones to match the corresponding

# size of the other array:

# a.shape -> (3, 3)

# b.shape -> (3, 3)

# Because the result matches, these shapes are compatible. We can see this here:

a+b
# Now let’s take a look at an example in which the two arrays are not compatible:



M = np.ones((3,2))

a = np.arange(3)



# This is just a slightly different situation than in the first example: the matrix M is

# transposed. How does this affect the calculation? The shapes of the arrays are:

# M.shape = (3, 2)

# a.shape = (3,)

# Again, rule 1 tells us that we must pad the shape of a with ones:

# M.shape -> (3, 2)

# a.shape -> (1, 3)

# By rule 2, the first dimension of a is stretched to match that of M :

# M.shape -> (3, 2)

# a.shape -> (3, 3)

# Now we hit rule 3—the final shapes do not match, so these two arrays are incompati‐

# ble, as we can observe by attempting this operation:



# print(M+a) #ERROR! operands could not be broadcast together with shapes
print(a[:, np.newaxis].shape)

M + a[:, np.newaxis]
x = np.array([1,2,3,4,5])



print(x<3)  # less than

print(x>3)  # greater than

print(x<=3) #less than or equal

print(x>=3) #greater than or equal

print(x!=3) #not equal

print(x==3) #equal
# It is also possible to do an element-by-element comparison of two arrays, and to

# include compound expressions:



(2*x) == (2**x)
# As in the case of arithmetic operators, the comparison operators are implemented as

# ufuncs in NumPy; for example, when you write x < 3 , internally NumPy uses

# np.less(x, 3) . A summary of the comparison operators and their equivalent ufunc

# is shown here:
rng = np.random.RandomState(seed=0)

x = rng.randint(10, size=(3,4))

print(x)



x<6
print(x)



# To count the number of True entries in a Boolean array, np.count_nonzero is useful:



# how many values less than 6?

print("1-: ",np.count_nonzero(x<6))



# We see that there are eight array entries that are less than 6. Another way to get at this

# information is to use np.sum ; in this case, False is interpreted as 0 , and True is inter‐

# preted as 1 :



print("2-: ",np.sum(x<6))



print("3-: ",np.sum(x!=np.nan))

print("4-: ",np.count_nonzero(x!=np.nan))
# how many values less than 6 in each row?

print(np.sum(x < 6, axis=1))



# how many values less than 6 in each column?

print(np.sum(x < 6, axis=0))
# If we’re interested in quickly checking whether any or all the values are true, we can

# use (you guessed it) np.any() or np.all() :



# are there any values greater than 8?

print(np.any(x>8))



# are there any values less than zero?

print(np.any(x<0))



# are all values less than 10?

print(np.all(x<10))



# are all values equal to 6?

print(np.all(x==6))
# are all values in each row less than 8?

print(np.all(x<8, axis=1))



# are all values in each column less than 3?

print(np.all(x<3, axis=0))
print(x)

print(x<5)

print(x[x<5])
# In Python, all nonzero integers will evaluate as True .

bool(42), bool(0), bool(-1)
bool(42 and 0)
bool(42 or 0)
# When you have an array of Boolean values in NumPy, this can be thought of as a

# string of bits where 1 = True and 0 = False , and the result of & and | operates in a

# similar manner as before:



A = np.array([1, 0, 1, 0, 1, 0], dtype=bool)

B = np.array([1, 1, 1, 0, 1, 1], dtype=bool)

A | B
x = np.arange(10)

(x > 4) & (x < 8)
import numpy as np



rand = np.random.RandomState(42)

x = rand.randint(100, size=10)

print(x)
[x[3],x[7],x[2]]
ind = [3,7,4]

x[ind]
ind = np.array([[3, 7],

                [4, 5]])

x[ind]
X = np.arange(12).reshape((3,4))

X
row = np.array([0,1,2])

col = np.array([2,1,3])

X[row,col]
X[row[:, np.newaxis], col] #row[:, np.newaxis].shape (3,1)
# Here, each row value is matched with each column vector, exactly as we saw in broad‐

# casting of arithmetic operations. For example:



row[:, np.newaxis] * col
print(X)
X[2,[2,0,1]]
X[1:, [2, 0, 1]]
mask = np.array([1, 0, 1, 0], dtype=bool)

X[row[:, np.newaxis], mask]  # mask 0 and 2 indixes are true!
x = np.arange(10)

i = np.array([2,1,8,4])

x[i] = 99

print(x)
x[i] -= 10

print(x)
x = np.zeros(10)

x[[0, 2]] = [4, 6]

print(x)
x = np.zeros(10)

x[[0, 0]] = [4, 6]

print(x)



# Where did the 4 go? The result of this operation is to first assign x[0] = 4 , followed

# by x[0] = 6 . The result, of course, is that x[0] contains the value 6.
i = [2, 3, 3, 4, 4, 4]

x[i] += 1

x
x = np.zeros(10)

np.add.at(x, i, 1)

print(x)
x = np.array([2,1,4,3,5])

np.sort(x)
x.sort()

print(x)
#return indices

x = np.array([2,1,4,3,5])

i = np.argsort(x)

print(i)



x[i]
# A useful feature of NumPy’s sorting algorithms is the ability to sort along specific

# rows or columns of a multidimensional array using the axis argument. For example:



rand = np.random.RandomState(42)

X = rand.randint(0,10,(4,6))

print(X)
# sort each column of X



np.sort(X, axis=0)
# sort each row of X



np.sort(X, axis=1)
# Note that the first three values in the resulting array are the three smallest in the

# array, and the remaining array positions contain the remaining values. Within the

# two partitions, the elements have arbitrary order.



x = np.array([7, 2, 1, 3, 6, 5, 4])

np.partition(x, 3)
# The result is an array where the first two slots in each row contain the smallest values

# from that row, with the remaining values filling the remaining slots.



np.partition(X, 2, axis=1)
np.partition(X, 2, axis=0)
np.argpartition(X, 2, axis=1)
np.argpartition(X, 2, axis=0)