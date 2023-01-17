import numpy as np 

# 1] Leverage arange and reshape method to create numpy array

a = np.arange(15).reshape(3,5)

print(a)



# Sample Output



# [[ 0  1  2  3  4]

# [ 5  6  7  8  9]

# [10 11 12 13 14]]

#2] Getting array shape

print(a.shape)

# (3,5)

#3]  Getting array dimension

print(a.ndim)

# 2

#4] Getting array size

print(a.size)

# 15
#5] Array with zeros

b = np.zeros((3,4))

print(b)



# [[ 0.  0.  0.  0.]

 #[ 0.  0.  0.  0.]

 #[ 0.  0.  0.  0.]]



#6] Array with ones. Three dimensional. Two rows. Each tows is 3X4 Matrix



c = np.ones((2,3,4))

print(c)



#[[[ 1.  1.  1.  1.]

 # [ 1.  1.  1.  1.]

 # [ 1.  1.  1.  1.]]



 #[[ 1.  1.  1.  1.]

  #[ 1.  1.  1.  1.]

  #[ 1.  1.  1.  1.]]]
#7] Array Indexing.

d = np.arange(10)



# Print Entire Array

print(d)

# [0 1 2 3 4 5 6 7 8 9]



#Print last element

print(d[-1])

#9



# Print 2nd and 3rd element

print(d[2:4])

# [2 3]



# Print from 2nd element to last but one

print(d[2:-1])

# [2 3 4 5 6 7 8]



# Multidimensional array indexing

e = np.arange(16).reshape(4,2,2)

# Print entire array

print(e)



#[[[ 0  1]

 # [ 2  3]]

#

# [[ 4  5]

#  [ 6  7]]

#

# [[ 8  9]

#  [10 11]]

#

# [[12 13]

#  [14 15]]]



# Print all rows from 1st. same as e[1:0]

print(e[1:])

#[[[ 4  5]

 # [ 6  7]]

#

 #[[ 8  9]

 # [10 11]]

#

 #[[12 13]

  #[14 15]]]



# Print 0th row. same as e[0:1]

print(e[:1])

#[[[0 1]

 # [2 3]]]