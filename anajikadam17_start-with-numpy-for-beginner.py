import numpy as np

a = np.array([3.2,4,6,5]) # array

print('\n',a)

b = np.array([1,4,2,5,3]) # integer array:

print('\n',b)
np.array([1,2,3,4], dtype="str")
np.array([3,6,2,3], dtype="float32")
# nested lists result in multidimensional arrays



np.array([range(i,i+3) for i in [2,4,6]])
# Create a 3x5 array filled with 5



np.full((3,5), 5)
# Create an array filled with a linear sequence

# Starting at 0, ending at 20, stepping by 2

# (this is similar to the built-in range() function)



np.arange(0,20,2)
x1 = np.array([5, 0, 3, 3, 7, 9])

x1
#To index from the start of the array, you can use positive indices:



x1[0]
#To index from the end of the array, you can use negative indices:



x1[-1]
import numpy as np

np.array
import numpy as np



# From list: 1d array

my_list = [10, 20, 30]

np.array(my_list)
# From list: 2d array

list_of_lists =  [[5, 10, 15], [20, 25, 30], [35, 40, 45]]

np.array(list_of_lists)
x = np.array([[3, 5, 2, 4],

       [7, 6, 8, 8],

       [1, 6, 7, 7]])

x
x[2,1]
x[2,-4]
x[-2,-3]
#You can also modify values using any of the above index notation:



x[0,0]=12

x
#Check data type of ndarray

type(np.array(x))
a =  np.array([[1, 2, 3,4,5,6],[7,8,9,10,11,12]]) 

print(a)

print(a.shape)
import numpy as np



np.arange(0, 10)
arr = np.array(my_list)



print(arr.dtype)
string = 'My_name'

print(string)
names = ['dumbledore', 'beeblebrox', 'skywalker', 'hermione', 'leia'] #list

names
names = ['dumbledore', 'beeblebrox', 'skywalker', 'hermione', 'leia']

capitalized_names = []

for name in names:

    capitalized_names.append(name.title())



# equals (do.. for)

capitalized_names = [name.title() for name in names]

capitalized_names
squares = [x**2 for x in range(9) if x % 2 == 0]  # square of Even integer 

print(squares)

# to add else statements, move the conditionals to the beginning

squares = [x**2 if x % 2 == 0 else x + 3 for x in range(9)]
location = (13.4125, 103.866667)   #Create Tuples

location
print("Latitude:", location[0])   #Access tuples

print("Longitude:", location[1])
# can also be used to assign multiple variables in a compact way

dimensions = 52, 40, 100    # tuple packing
# tuple unpacking

length, width, height = dimensions

print("The dimensions are {} x {} x {}".format(length, width, height))
numbers = [1, 2, 6, 3, 1, 1, 6]   #list convert to set

unique_nums = set(numbers)

print(unique_nums) # {1, 2, 3, 6}
fruit = {"apple", "banana", "orange", "grapefruit"} #create set

print(fruit)  #print set

print("watermelon" in fruit)  #check element in set

fruit.add("watermelon")   #add element in set

print(fruit)   #after adding print set

print("watermelon" in fruit)  #check element in set
print(fruit.pop())   #remove a random element from set

print(fruit)
elements={}       # Create empty dictionary

elements = {"hydrogen": 1, "helium": 2, "carbon": 6}       #create dict

elements
print(elements["helium"])  #accing element
elements["lithium"] = 3   #addind element

elements
# Just keys

for key in elements:

    print(key)

# Keys and values

for key, value in elements.items():

    print("Actor: {}    Role: {}".format(key, value))
del elements['carbon']  #delete value from dict

elements

#alternative way

elements.pop('helium')

elements
# check whether a value is in a dictionary, the same way we check whether a value is in a list or set with the in keyword.

print("carbon" in elements) # True
# get() looks up values in a dictionary, but unlike square brackets, get returns None (or a default value of your choice) if the key isn't found.

# If you expect lookups to sometimes fail, get might be a better tool than normal square bracket lookups.

print(elements.get("dilithium")) # None

print(elements.get('kryptonite', 'There\'s no such element!'))

# "There's no such element!"
n = elements.get("dilithium")

print(n is None) # True

print(n is not None) # False
a = [1, 2, 3]

b = a

c = [1, 2, 3]

print(a == b) # True

print(a is b) # True

print(a == c) # True

print(a is c) # False

# List a and list b are equal and identical.

# List c is equal to a (and b for that matter) since they have the same contents. But a and c (and b for that matter, again) point to two different objects, i.e., they aren't identical objects.

# That is the difference between checking for equality vs. identity.
elements = {"hydrogen": {"number": 1,

                         "weight": 1.00794,

                         "symbol": "H"},

              "helium": {"number": 2,

                         "weight": 4.002602,

                         "symbol": "He"}}

helium = elements["helium"]  # get the helium dictionary

hydrogen_weight = elements["hydrogen"]["weight"]  # get hydrogen's weight

oxygen = {"number":8,"weight":15.999,"symbol":"O"}  # create a new oxygen dictionary 

elements["oxygen"] = oxygen  # assign 'oxygen' as a key to the elements dictionary

print('elements = ', elements)
words =  ['great', 'expectations','the', 'adventures', 'of', 'sherlock','holmes','the','great','gasby','hamlet','adventures','of','huckleberry','fin'];

word_counter = {}

for word in words:

    word_counter[word] = word_counter.get(word,0)+1;

print(word_counter);
z = np.zeros((2, 3))

print('Zeros: \n',z)



o = np.ones((2, 4))

print('Ones: \n',o)



e = np.eye(3)

print('Eye: \n',e)

# divide into 7 interval from 0 to 10

ls  =  np.linspace(0, 10, 7)

print('linespace:\n',ls)



# 2 x 3 ndarray full of fives

# np.full(shape, constant value)

X = np.full((2,3), 5)

print('X:\n',X)
# random number (uniform distribution) array of shape (3 , 4)



a = np.random.rand(3, 4)

print('a:\n',a)

# random number (standard normal distribution) array of shape (2, 3)



b = np.random.randn(2, 3)

print ('b:\n',b)



# 10 random integers between 4 (inclusive) to 40 (exclusive)



c = np.random.randint(4, 40, 10)

print('c:\n',c)

# 10 random integers upto 50 (exclusive). This makes the start value default to 0.

# The size parameter dictates the return array shape



d = np.random.randint(50, size=(4,4))

print('d:\n',d)
a1= np.arange(1, 10, dtype=np.float16).reshape(3, 3)

a2 = np.arange(100, 109, dtype=np.float16).reshape(3, 3)



print('a1:',a1)

print('a2:',a2)



print('Vector Division:\n',a2 / a1)  # Vector Division

print('Scalar Division:\n',a2 / 3)    # Scalar Division



ab= np.arange(1, 10, dtype=np.float16).reshape(3, 3)

abc = a1 > 5   #comparison 



print(ab)

print(abc)



print(type(ab))

print(abc.dtype)
x = np.array([1,2,3,4,5])



print(x<3)  # less than

print(x>3)  # greater than

print(x<=3) #less than or equal

print(x>=3) #greater than or equal

print(x!=3) #not equal

print(x==3) #equal
# In Python, all nonzero integers will evaluate as True .

bool(42), bool(0), bool(-1)
bool(42 and 0)
bool(42 or 0)
# When you have an array of Boolean values in NumPy, this can be thought of as a

# string of bits where 1 = True and 0 = False , and the result of  | operates in a

# similar manner as before:



A = np.array([1, 0, 1, 0, 1, 0], dtype=bool)

B = np.array([1, 1, 1, 0, 1, 1], dtype=bool)

A | B
# When you have an array of Boolean values in NumPy, this can be thought of as a

# string of bits where 1 = True and 0 = False , and the result of & operates in a

# similar manner as before:



A = np.array([1, 0, 1, 0, 1, 0], dtype=bool)

B = np.array([1, 1, 1, 0, 1, 1], dtype=bool)

A & B
# Find max, min, mean of given ndarray

my_list = [10, 20, 30, 40, 50, 60, 70, 80, 90]

temp = np.array(my_list)

print('Max: ', temp.max())

print('Min: ', temp.min())

print('Mean: ', temp.mean())



# Find index of max/min 

print('Argmax: ', temp.argmax())

print('Argmin: ', temp.argmin())
# As a quick example, consider computing the sum of all values in an array. Python

# itself can do this using the built-in sum function:



L = np.random.random(100)

sum(L)
#The syntax is quite similar to that of NumPy’s sum function, and the result is the same

#in the simplest case:



np.sum(L)
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

x = np.array([[5.3, 10.2, 15.1], [20.4, 25.3, 30.9], [35.4, 40.1,45.6]])

print(x.dtype)

print(x.shape)
# Another useful type of operation is reshaping of arrays. The most flexible way of

# doing this is with the reshape() method. For example, if you want to put the numbers

# 1 through 9 in a 3×3 grid, you can do the following:



grid = np.arange(1,10,1).reshape(3,3)

print(grid)
x = np.random.randn(4,3)

x.reshape(3,4)
x = np.arange(10)

x
x[:5] # first five elements
x[5:] # elements after index 5
x[4:7]# middle subarray
x[::2] # every other element

x[1::2] #every other element, starting at index 1
x[-7:-2:2]
# A potentially confusing case is when the step value is negative. In this case, the

# defaults for start and stop are swapped. This becomes a convenient way to reverse

# an array:



x[::-1] # all elements, reversed
x[5::-2]# reversed every other from index 5
x[5:-8:-1]
z = np.array([[12,  5,  2,  4],

       [ 7,  6,  8,  8],

       [ 1,  6,  7,  7]])

z
# two rows, three columns



z[:2, :3]
# all rows, every other column



z[:3,::2]
#Finally, subarray dimensions can even be reversed together:



z[::-1,::-1]
print(z[:, 0]) # first column of z
print(z[0,:]) # first row of z
#In the case of row access, the empty slice can be omitted for a more compact syntax:



print(z[0]) # equivalent to z[0, :]
#Let’s extract a 2×2 subarray from this:



x2_sub = z[:2,:2]

print(x2_sub)
#Now if we modify this subarray, we’ll see that the original array is changed! Observe:



x2_sub[0,0] = 99

print(x2_sub)

print(z)
x2_sub_copy = z[:2,:2].copy()

print(x2_sub_copy)
#If we now modify this subarray, the original array is not touched:



x2_sub_copy[0,0] = 42

print(x2_sub_copy)
x = np.array([1,2,3])

y = np.array([3,2,1])

np.concatenate((x, y))

#You can also concatenate more than two arrays at once:

z = np.array([99,99,99]) 





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
# indexing one dimensional array

import numpy as np



arr = np.arange(10)

print("Array:", arr)



# get the element at index 5

print("Element:", arr[5])



#Get values in a range

print("Slice:", arr[1:9:2])
# indexing two dimensional array

arr = np.array([[1,2,3],[4,5,6],[7,8,9]])

                 #0      #1       #2

print (arr)

print (arr[1])    # select a row

print (arr[2][2]) #[row], [column]

print (arr[0,2])  # [row, column] 
# Slicing one dimensional array

arr = np.arange(10)

print (arr)



print (arr[0:3])



# start from first index and get every 3rd elemnt

print (arr[1::3])
# Slicing two-dimensional array



arr = np.array([[1, 2, 3, 4, 5],

                [6, 7, 8, 9, 10],

                [11,12,13,14,15]])





# 1st row to 2nd row , all columns

print(arr[1:3, 1:4])



# notice that the output is also a 2d array
arr = np.array([[1,2,3],[4,5,6],[7,8,9]])



below5_filter = (arr < 5)

print(below5_filter)

print(arr[below5_filter])
# Replacing Values

import numpy as np



vector = np.array([5, 10, 15, 20])

print (vector)



equal_to_ten_or_five = (vector == 10) | (vector == 5)

vector[equal_to_ten_or_five] = 50



print (vector)
# Delete

# np.delete(ndarray, elements, axis)

x = np.array([1, 2, 3, 4, 5])

# delete the first and fifth element of x

x = np.delete(x, [0,4])

print(x)

Y = np.array([[1,2,3],[4,5,6],[7,8,9]])

# delete the first row of Y

w = np.delete(Y, 0, axis=0)

print(w)

# delete the first and last column of Y

v = np.delete(Y, [0,2], axis=1)

print(v)
# Append

# np.append(ndarray, elements, axis)

# append the integer 6 to x

x = np.append(x, 6)

print(x)

# append the integer 7 and 8 to x

x = np.append(x, [7,8])

print(x)

# append a new row containing 7,8,9 to y

v = np.append(Y, [[10,11,12]], axis=0)

print(v)

# append a new column containing 9 and 10 to y

q = np.append(Y,[[13],[14],[15]], axis=1)

print(q)
# Insert

# np.insert(ndarray, index, elements, axis)

# inserts the given list of elements to ndarray right before

# the given index along the specified axis

x = np.array([1, 2, 5, 6, 7])

print('x:\n',x)

Y = np.array([[1,2,3],[7,8,9]])

print('y:\n',y)

# insert the integer 3 and 4 between 2 and 5 in x. 

x = np.insert(x,2,[3,4])

print('x:\n',x)

# insert a row between the first and last row of Y

w = np.insert(Y,1,[4,5,6],axis=0)

print('w:\n',w)

# insert a column full of 5s between the first and second column of Y

v = np.insert(Y,1,5, axis=1)

print('v:\n',v)
# Stacking

# NumPy also allows us to stack ndarrays on top of each other,

# or to stack them side by side. The stacking is done using either

# the np.vstack() function for vertical stacking, or the np.hstack()

# function for horizontal stacking. It is important to note that in

# order to stack ndarrays, the shape of the ndarrays must match.

x = np.array([1,2])

print('x:\n',x)

Y = np.array([[3,4],[5,6]])

print('y:\n',Y)

z = np.vstack((x,Y)) # [[1,2], [3,4], [5,6]]

print('z:\n',z)

w = np.hstack((Y,x.reshape(2,1))) # [[3,4,1], [5,6,2]]

print('w:\n',w)
# Copy

# if we want to create a new ndarray that contains a copy of the

# values in the slice we need to use the np.copy()

# create a copy of the slice using the np.copy() function

Z = np.copy(X[1:4,2:5])

print(Z)

#  create a copy of the slice using the copy as a method

W = X[1:4,2:5].copy()

print(W)
# Extract elements along the diagonal

d0 = np.diag(X)

# As default is k=0, which refers to the main diagonal.

# Values of k > 0 are used to select elements in diagonals above

# the main diagonal, and values of k < 0 are used to select elements

# in diagonals below the main diagonal.

d1 = np.diag(X, k=1)

print(d1)

d2 = np.diag(X, k=-1)

print(d2)
#Find Unique Elements in ndarray

X = [1,2,3,4,5,5,6,6,6,7,8,9,9,9]

u = np.unique(X)

u
# Boolean Indexing

X = np.arange(25).reshape(5, 5)

print('The elements in X that are greater than 10:', X[X > 10])

print('The elements in X that less than or equal to 7:', X[X <= 7])

print('The elements in X that are between 10 and 17:', X[(X > 10) & (X < 17)])



# use Boolean indexing to assign the elements that

# are between 10 and 17 the value of -1

X[(X > 10) & (X < 17)] = -1
# Set Operations

x = np.array([1,2,3,4,5])

y = np.array([6,7,2,8,4])

print('The elements that are both in x and y:', np.intersect1d(x,y))

print('The elements that are in x that are not in y:', np.setdiff1d(x,y))

print('All the elements of x and y:',np.union1d(x,y))
# Sorting

# When used as a function, it doesn't change the original ndarray

s = np.sort(x)

print(s)

# When used as a method, the original array will be sorted

x.sort()

print(x)



# sort x but only keep the unique elements in x

s = np.sort(np.unique(x))



# sort the columns of X

s = np.sort(X, axis = 0)



# sort the rows of X

s = np.sort(X, axis = 1)
#Math Functions

# NumPy allows element-wise operations on ndarrays as well as

# matrix operations. In order to do element-wise operations,

# NumPy sometimes uses something called Broadcasting.

# Broadcasting is the term used to describe how NumPy handles

# element-wise arithmetic operations with ndarrays of different shapes.

# For example, broadcasting is used implicitly when doing arithmetic

# operations between scalars and ndarrays.

x = np.array([1,2,3,4])

y = np.array([5.5,6.5,7.5,8.5])

print(np.add(x,y))

print(np.subtract(x,y))

print(np.multiply(x,y))

print(np.divide(x,y))



# in order to do these operations the shapes of the ndarrays

# being operated on, must have the same shape or be broadcastable

X = np.array([1,2,3,4]).reshape(2,2)

print(X)

Y = np.array([5.5,6.5,7.5,8.5]).reshape(2,2)

print(Y)

print(np.add(X,Y))

print(np.subtract(X,Y))

print(np.multiply(X,Y))

print(np.divide(X,Y))



# apply mathematical functions to all elements of an ndarray at once

print(np.exp(x))

print(np.sqrt(x))

print(np.power(x,2))
# Statistical Functions

print('Average of all elements in X:', X.mean())

print('Average of all elements in the columns of X:', X.mean(axis=0))

print('Average of all elements in the rows of X:', X.mean(axis=1))

print()

print('Sum of all elements in X:', X.sum())

print('Standard Deviation of all elements in X:', X.std())

print('Median of all elements in X:', np.median(X))

print('Maximum value of all elements in X:', X.max())

print('Minimum value of all elements in X:', X.min())
# Broadcasting

# NumPy is working behind the scenes to broadcast 3 along the ndarray

# so that they have the same shape. This allows us to add 3 to each

# element of X with just one line of code.

print(4*X)

print(4+X)

print(4-X)

print(4/X)

# NumPy is able to add 1 x 3 and 3 x 1 ndarrays to 3 x 3 ndarrays

# by broadcasting the smaller ndarrays along the big ndarray so that

# they have compatible shapes. In general, NumPy can do this provided

# that the smaller ndarray can be expanded to the shape of the larger

# ndarray in such a way that the resulting broadcast is unambiguous.

x = np.array([1,2,3])

Y = np.array([[1,2,3],[4,5,6],[7,8,9]])

Z = np.array([1,2,3]).reshape(3,1)

print(x + Y)

print(Z + Y)