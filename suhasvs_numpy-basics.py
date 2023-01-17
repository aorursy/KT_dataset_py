import numpy as np

a = np.array([1,2,3])# Create a rank 1 array with elements 1,2,3

print("Type of array a is",type(a));
print("Shape of array a is",a.shape)# print its type and shape

for item in range(0,len(a)):
    print(a[item])           # Print each of its element individually

a[0]=5# Change element of the array at zero index with 5

print("The new array of a is",a) # print new array

b = np.array([[1,2,3],[4,5,6]]) # Create a rank 2 array like [[1,2,3],[4,5,6]]

print(b.shape)# print its shape

print(b[0, 0], b[0, 1], b[1, 0])   # Prints "1 2 4"
a = np.array([[0,0],[0,0]])# Create an array of all zeros with shape (2,2)
print(a) 
print("..............")# added seperator

b = np.array([1,1])# Create an array of all ones with shape (1,2)
print(b)           
print("..............")# added seperator

c =  np.array([[7,7],[7,7]])# Create a constant array with shape (2,2) and value 7
print(c);print(c.shape)              
print("..............")# added seperator

d =np.array([[1,0],[0,1]])# Create a 2x2 identity matrix
print(d)              
print("..............")# added seperator

e=np.array([[4,18],[-9,45]])# Create an (2,2) array filled with random values
print(e)
print("..............")# added seperator
# Create the following rank 2 array with shape (3, 4) and values as
# [[ 1  2  3  4]
#  [ 5  6  7  8]
#  [ 9 10 11 12]]
a = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
print("Array a is:\n",a)
print("..............")# added seperator

# Use slicing to pull out the subarray consisting of the first 2 rows
# and columns 1 and 2; b is the following array of shape (2, 2):
b = a[0:2,1:3]
print("Sliced array b is:\n",b)
print("..............")# added seperator

# A slice of an array is a view into the same data, so modifying it
# will modify the original array.
print(a[0, 1])   # Prints "2"
b[0, 0] = 77     # b[0, 0] is the same piece of data as a[0, 1]
print(a[0, 1])   # Prints "77"
# Create the following rank 2 array with shape (3, 4) as
# [[ 1  2  3  4]
#  [ 5  6  7  8]
#  [ 9 10 11 12]]
a = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
print("Array a is:\n",a)
print("..............",a.ndim)# added seperator

# Two ways of accessing the data in the middle row of the array.
# Mixing integer indexing with slices yields an array of lower rank,
# while using only slices yields an array of the same rank as the
# original array:

row_r1 = a[1] # Create Rank 1 view of the second row of a
row_r2 = np.array([a[1]])# Create Rank 2 view of the second row of a
print(row_r1, row_r1.shape,"rank is",row_r1.ndim)
print(row_r2, row_r2.shape,"rank is",row_r2.ndim)
 
# We can make the same distinction when accessing columns of an array:
 
col_r1 = a[:,1]# Create Rank 1 view of the second column of a
col_r2 = np.array([a[:,1]])# Create Rank 2 view of the second column of a
print(col_r1, col_r1.shape,"rank is",col_r1.ndim) 
print(col_r2, col_r2.shape,"rank is",col_r2.ndim) 
a = np.array([[1,2], [3, 4], [5, 6]])

# An example of integer array indexing.
# Using the array "a" created above and integer indexing
# print an array  that should have shape (3,) and should print "[1 4 5]" when printed
print(np.array([a[0, 0], a[1, 1], a[2, 0]]))
print(a.shape)

# The method of integer array indexing that you implemented is equivalent to this:
print(np.array([a[0, 0], a[1, 1], a[2, 0]]))  # Prints "[1 4 5]"

# When using integer array indexing, you can reuse the same
# element from the source array:
print(a[[0, 0], [1, 1]])  # Prints "[2 2]"

# Equivalent to the previous integer array indexing example
print(np.array([a[0, 1], a[0, 1]]))  # Prints "[2 2]"
# Create a new array from which we will select elements
a = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
print(a) 


# Create an array of indices
b = np.array([0, 2, 0, 1])
print(b)
# Select one element from each row of a using the indices in b and print it

print(a[b[0]],a[b[2]],a[b[0]],a[b[1]])
# Mutate one element from each row of a using the indices in b by adding 10 to it

print(a)
a = np.array([[1,2], [3, 4], [5, 6]])
print(a)
bool_idx = a > 2# Find the elements of a that are bigger than 2 using boolean indexing;

print(bool_idx)     

# We use boolean array indexing to construct a rank 1 array
# consisting of the elements of a corresponding to the True values
# of bool_idx
print(a[bool_idx])  

# We can do all of the above in a single concise statement:
print(a[a > 2])    
x = np.array([1, 2])   # Let numpy choose the datatype
print(x.dtype)         # Prints "int64"

x = np.array([1.0, 2.0])   # Let numpy choose the datatype
print(x.dtype)             # Prints "float64"

x = np.array([1, 2], dtype=np.int64)   # Force a particular datatype
print(x.dtype)                         # Prints "int64"
x = np.array([[1,2],[3,4]], dtype=np.float64)
y = np.array([[5,6],[7,8]], dtype=np.float64)

# print Elementwise sum of x and y
print("array x\n",x)
print("array y\n",y)

print("print Elementwise sum of x and y")
print(x+y)

# print Elementwise difference of x and y
print("print Elementwise difference of x and y")
print(x-y)

# print Elementwise product of x and y
print("print Elementwise product of x and y")
print(x*y)

# print Elementwise divison of x and y
print("print Elementwise divison of x and y")
print(x/y)

# print Elementwise square root of x
print("print Elementwise square root of x")
print(x**0.5)

x = np.array([[1,2],[3,4]])
y = np.array([[5,6],[7,8]])

v = np.array([9,10])
w = np.array([11, 12])

# print Inner product of vectors v and w;
print(np.dot(v,w))
print()

# print Matrix / vector product of x and v; 
print(x.dot(v))
print()

# print Matrix / matrix product of x and y;
print(np.matmul(x,y))
print()
x = np.array([[1,2],[3,4]])

print("Array x is\n",x)
# Compute sum of all elements and print it
print("sum of all elements in x",np.sum(x))

# Compute sum of each column and print it
print("sum of each column",np.sum(x,axis=0))

# Compute sum of each row and print it
print("sum of each row",np.sum(x,axis=1))
x = np.array([[1,2], [3,4]])
print("Array x is\n",x)
# print x and its transpose
print("Transpose of array x\n",np.transpose(x))
# Note that taking the transpose of a rank 1 array does nothing:
v = np.array([1,2,3])
print("Array v is\n",v)
print("Transpose of array v\n",np.transpose(v))
# print v and its transpose

# You will add the vector v to each row of the matrix x,
# storing the result in the matrix y
x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
v = np.array([1, 0, 1])
y = np.empty_like(x)   # Create an empty matrix with the same shape as x
y= x+v
# Add the vector v to each row of the matrix x with an explicit loop and store in y


print(y)
# You will add the vector v to each row of the matrix x,
# storing the result in the matrix y

x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
v = np.array([1, 0, 1])

y=x+v

vv = np.tile(v,4)# Stack 4 copies of v on top of each other using tile method
print(vv)                 


#y = x+vv # Add x and vv elementwise
print(y)  


# You will add the vector v to each row of the matrix x,
# storing the result in the matrix y
x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
v = np.array([1, 0, 1])
y = x+v# Add v to each row of x using broadcasting
print(y) 
# Compute outer product of vectorsnp.transpose
v = np.array([1,2,3])  # v has shape (3,)
w = np.array([4,5])    # w has shape (2,)

print("Outer product of v and w",np.outer(v,w))# Compute an outer product of v and w and print it


# Add a vector to each row of a matrix
x = np.array([[1,2,3], [4,5,6]])
# Add v to ecah row of x and print it
print(x+v)
# Add a vector to each column of a matrix
# Add vector w to each column of x and print it
#print(w+x)

# Another solution is to reshape w to be a column vector of shape (2, 1);
# we can then broadcast it directly against x to produce the same
# output.
print(x + np.reshape(w, (2, 1)))

# Multiply a matrix by a constant:
# x has shape (2, 3). Numpy treats scalars as arrays of shape ();
# these can be broadcast together to shape (2, 3), producing the
# following array:
# [[ 2  4  6]
#  [ 8 10 12]]
print(x * 2)