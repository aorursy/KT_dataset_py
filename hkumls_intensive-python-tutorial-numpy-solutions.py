import numpy as np      #import numpy library, use np as a shorter name

a = np.array([[1, 3, 5],[6, 4, 5]])   # Create a rank 1 array
print(a.shape)            # Prints "(2,3), which represents the 2 rows and 3 columns
a                        #Shows array a
print(a[0,1], a[1,0])   # Prints the elements from the first row, second column 
                        # and second row, first column of the array
a[0,1] = 5                  # Change an element of the array
print(a)                  # Prints "[[1, 3, 5], [6, 4, 5]]"
b = np.array([[1,2,3],[4,5,6]])           # Create a rank 2 array
### START CODE HERE ### (≈ 2 line of code)
print(b.shape)                      # Fill a function to return the shape of the numpy array
b[0, :2] = 10                       # Change the third column elements into 10
### END CODE HERE ###
print(b)
import numpy as np

a = np.zeros((2,2))   # Create an array of all zeros
print(a)              # Prints "[[ 0.  0.]
                      #          [ 0.  0.]]"

b = np.ones((1,2))    # Create an array of all ones
print(b)              # Prints "[[ 1.  1.]]"

c = np.full((2,2), 7)  # Create a constant array
print(c)               # Prints "[[ 7.  7.]
                       #          [ 7.  7.]]"

d = np.eye(2)         # Create a 2x2 identity matrix
print(d)              # Prints "[[ 1.  0.]
                      #          [ 0.  1.]]"

e = np.random.random((2,2))  # Create an array filled with random values
print(e)                     # Might print "[[ 0.91940167  0.08143941]
                             #               [ 0.68744134  0.87236687]]"
    
f = np.linspace(1, 15, 3)   # The linspace() function returns numbers evenly spaced over a specified intervals.
print (f)                   # Say we want 3 evenly spaced points from 1 to 15, we can easily use this.
                            # linspace() takes the third argument as the number of datapoints to be created
g= np.arange(3,10)       # Lists the natural numbers from 3 to 9, as the number in the second position is excluded
print (g)
### START CODE HERE ### (≈ 2 line of code)
Z = np.eye(3)            
W = np.zeros(3)
### END CODE HERE ###
print(Z)
print(W)
# Create the following rank 2 array with shape (3, 4)
# [[ 1  2  3  4]
#  [ 5  6  7  8]
#  [ 9 10 11 12]]
a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
a
# Use slicing to pull out the subarray consisting of the first 2 rows
# and columns 1 and 2; b is the following array of shape (2, 2):
# [[2 3]
#  [6 7]]
b = a[:2, 1:3]
print (b)
# A slice of an array is a view into the same data, so modifying it
# will modify the original array.
print(a[0, 1])   # Prints the element from the first row, second column, which is "2" 
b[0, 0] = 77     # b[0, 0] is the same piece of data as a[0, 1], 
                 # as we change the value of the corresponding position to 77 for both the a and b array
print(a[0, 1])   # Prints "77"

# Create a 2d array with 1 on the border and 0 inside
# Hint: make a numpy array full of ones, and then replace the selected elements into zero

Z = np.ones((5,5))
### START CODE HERE ### (≈ 1 line of code)
Z[1:-1,1:-1] = 0                        # How should we change the interior part of the array into zero?
### END CODE HERE ###
print(Z)
import numpy as np

a = np.array([[1,2], [3, 4], [5, 6]])    #Think, what does this array look like? 

# An example of integer array indexing.
# The returned array will have shape (3, 2) 

print(a[[0, 1, 2], [0, 1, 0]])  # Prints the element in position [0,0], [1,1], [2,0] - which is [1 4 5]

# When using integer array indexing, you can reuse the same
# element from the source array:
print(a[[0, 0], [1, 1]])  # Prints the element in position [0,1] twice - which is [2 2]

import numpy as np

a = np.array([[1,2], [3, 2], [5, 1]])

bool_idx = (a > 2)   # Find the elements of a that are bigger than 2;
                     # this returns a numpy array of Booleans of the same
                     # shape as a, where each slot of bool_idx tells
                     # whether that element of a is > 2.

print(bool_idx)      # Prints "[[False False]
                     #          [ True  False]
                     #          [ True  False]]"

# We use boolean array indexing to construct a rank 1 array 
# consisting of the elements of a corresponding to the True values
# of bool_idx
print(a[bool_idx])  # Prints "[3 5]", since these elements are True values

# We can also do all of the above in a single concise statement:
print(a[a > 2])     # Prints "[3 5]"
x = np.array([[1,2], [13, 2], [12, 1]])
### START CODE HERE ### (≈ 1 line of code)
y = x[x > 6]
### END CODE HERE ###
print(y)
import numpy as np

x = np.array([1, 2])   # Let numpy choose the datatype
print(x.dtype)         # Prints "int64"

x = np.array([1.0, 2.0])   # Let numpy choose the datatype
print(x.dtype)             # Prints "float64"

x = np.array([1, 2], dtype=np.int64)   # Force a particular datatype
print(x.dtype)                         # Prints "int64"
arr = np.arange(1,11)          # Numbers from 1 to 10
print (arr)
print (arr * arr)              # Multiplies each element by itself 
print (arr - arr)              # Subtracts each element from itself
print (arr + arr)              # Adds each element to itself
print (arr / arr)              # Divides each element by itself
#Another example for mathematical operations in numpy

x = np.array([[1,2],[3,4]], dtype=np.float64)
y = np.array([[5,6],[7,8]], dtype=np.float64)

# Elementwise sum; both produce the array
# [[ 6.0  8.0]
#  [10.0 12.0]]
print(np.add(x, y))

# Elementwise difference; both produce the array
# [[-4.0 -4.0]
#  [-4.0 -4.0]]
print(np.subtract(x, y))

# Elementwise product; both produce the array
# [[ 5.0 12.0]
#  [21.0 32.0]]
print(np.multiply(x, y))

# Elementwise division; both produce the array
# [[ 0.2         0.33333333]
#  [ 0.42857143  0.5       ]]
print(np.divide(x, y))

# Elementwise square root; produces the array
# [[ 1.          1.41421356]
#  [ 1.73205081  2.        ]]
print(np.sqrt(x))
x = np.array([[1,2],[3,4]])
### START CODE HERE ### (≈ 1 line of code)
x = np.exp(x)                
### END CODE HERE ###
print(x)
# GRADED FUNCTION: basic_sigmoid

import math

def basic_sigmoid(x):
    """
    Compute sigmoid of x.

    Arguments:
    x -- A scalar

    Return:
    s -- sigmoid(x)
    """
    
    ### START CODE HERE ### (≈ 1 line of code)
    s = 1/(1+math.exp(-x))
    ### END CODE HERE ###
    
    return s
basic_sigmoid(3)
### One reason why we use "numpy" instead of "math" in Deep Learning ###
x = [1, 2, 3]
basic_sigmoid(x) # you will see this give an error when you run it, because x is a vector.
# example of vector operation
x = np.array([1, 2, 3])
print (x + 3)
# GRADED FUNCTION: sigmoid

import numpy as np # this means you can access numpy functions by writing np.function() instead of numpy.function()

def sigmoid(x):
    """
    Compute the sigmoid of x

    Arguments:
    x -- A scalar or numpy array of any size

    Return:
    s -- sigmoid(x)
    """
    
    ### START CODE HERE ### (≈ 1 line of code)
    s = 1 / (1 + np.exp(-x))
    ### END CODE HERE ###
    
    return s
x = np.array([1, 2, 3])
sigmoid(x)
# GRADED FUNCTION: sigmoid_derivative

def sigmoid_derivative(x):
    """
    Compute the gradient (also called the slope or derivative) of the sigmoid function with respect to its input x.
    You can store the output of the sigmoid function into variables and then use it to calculate the gradient.
    
    Arguments:
    x -- A scalar or numpy array

    Return:
    ds -- Your computed gradient.
    """
    
    ### START CODE HERE ### (≈ 2 lines of code)
    s = 1 / (1 + np.exp(-x))
    ds = s * (1 - s)
    ### END CODE HERE ###
    
    return ds
x = np.array([1, 2, 3])
print ("sigmoid_derivative(x) = " + str(sigmoid_derivative(x)))
import numpy as np

x = np.array([[1,2],[3,4]])  # What does this array look like?

print(np.sum(x))  # Compute sum of all elements; prints "10"
print(np.sum(x, axis=0))  # Compute sum of each column; prints "[4 6]"
print(np.sum(x, axis=1))  # Compute sum of each row; prints "[3 7]"
# GRADED FUNCTION: L1

def L1(yhat, y):
    """
    Arguments:
    yhat -- vector of size m (predicted labels)
    y -- vector of size m (true labels)
    
    Returns:
    loss -- the value of the L1 loss function defined above
    """
    
    ### START CODE HERE ### (≈ 1 line of code)
    loss = np.sum(np.abs(y - yhat))
    ### END CODE HERE ###
    
    return loss
yhat = np.array([.9, 0.2, 0.1, .4, .9])
y = np.array([1, 0, 0, 1, 1])
print("L1 = " + str(L1(yhat,y)))
# GRADED FUNCTION: L2

def L2(yhat, y):
    """
    Arguments:
    yhat -- vector of size m (predicted labels)
    y -- vector of size m (true labels)
    
    Returns:
    loss -- the value of the L2 loss function defined above
    """
    
    ### START CODE HERE ### (≈ 1 line of code)
    loss = np.sum (np.power(y-yhat,2))
    ### END CODE HERE ###
    
    return loss
yhat = np.array([.9, 0.2, 0.1, .4, .9])
y = np.array([1, 0, 0, 1, 1])
print("L2 = " + str(L2(yhat,y)))