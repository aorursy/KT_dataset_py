import numpy as np

import matplotlib.pyplot as plt
no_of_items = 10

lower_limit= 5

upper_limit= 25
np.random.seed(0)



values = np.random.randint(lower_limit, upper_limit, no_of_items)

values
no_of_rows = 5

no_of_columns = 2

containers = values.reshape(no_of_rows, no_of_columns)
containers
radius = containers[:,0]
radius
height = containers[:,1]
height
valume = np.pi*(radius**2)*height
valume
total_valume = valume.sum()
total_valume
# Create an array with 10^7 elements.

arr = np.arange(1e7)
# Converting ndarray to list

larr = arr.tolist()
larr
def  list_times(alist, scalar):

    for i, val in enumerate(alist):

        alist[i] = val*scalar

    return alist
#Using IPython's magic timeit command

%timeit arr * 1.1
%timeit list_times(larr, 1.1)
# Creating a 3D numpy array



arr = np.zeros((3,3,3))



print(arr)

# Trying to convert array to a matrix, witch will not work



#mat = np.matrix(arr)



#print(mat)

#ValueError: Shape too large to be a matrix 



# First we create a list and then

# wrap it with the np.array() function .



alist=[1,2,3]



arr = np.array(alist)



print(arr)



# Creating an array of zeros with five elements



arr = np.zeros(5)

print(arr)

#Create an array from 0 to 100

arr = np.arange(100)

arr
# Creating array from 10 to 100

arr = np.arange(10, 100)

print(arr)
# Want 100 steps from 0 to 1

arr = np.linspace(0, 1, 100)

print(arr)
# Want to generate an array from 1 to 10

# in log10 space in 100 steps ..

arr = np.logspace(0, 1, 100, base=10.0)

print(arr)
print(np.log10(2))
### creating a 5x5 array of zeros (an image)



image = np.zeros((5,5))

image
img = np.array([[ 0.,  0.,  5., 13.,  9.,  1.,  0.,  0.],

       [ 0.,  0., 13., 15., 10., 15.,  5.,  0.],

       [ 0.,  3., 15.,  2.,  0., 11.,  8.,  0.],

       [ 0.,  4., 12.,  0.,  0.,  8.,  8.,  0.],

       [ 0.,  5.,  8.,  0.,  0.,  9.,  8.,  0.],

       [ 0.,  4., 11.,  0.,  1., 12.,  7.,  0.],

       [ 0.,  2., 14.,  5., 10., 12.,  0.,  0.],

       [ 0.,  0.,  6., 13., 10.,  0.,  0.,  0.]])

plt.imshow(img)

plt.show()
plt.imshow(image)

plt.show()
# 16-bit floating -point

cube = np.ones((5,5,5)).astype(np.float16)

cube
# Array fo zero integers

arr = np.zeros(2, dtype=int)

arr
#array of zero floats

arr = np.zeros(2, dtype=np.float32)

arr
# Creating an array with elements from 0 to 999

arr1d = np.arange(1000)



#Now reshaping the array to a 10x10x10 3D array

arr3d = arr1d.reshape((10,10,10))



# The reshape command can alternatively be called this way

arr3d = np.reshape(arr1d, (10,10,10))



# Inversely, we can flatten arrays

arr4d = np.zeros((10,10, 10, 10))



#flatten Arrays

arr1d = arr4d.ravel()
#test Flatten arrays

arr2d = np.array([[4, 5],[8, 6]])

print(arr2d)

print(arr2d.ravel())
# Creating an array of zeros and defining column types



recarr = np.zeros((2,), dtype=('i4,f4,a10'))



toadd = [(1,2.3, 'hello'), (2,3.,"world")]

recarr[:] = toadd

recarr
alist =np.array([[1,2],[3,4]])

print(alist)
print(alist[0][1])

print(alist[1][1])
print(alist[:])
print(alist[:,1])
print(alist[:,0])
print(alist[1,:])
#arr = np.arange(5)

arr  = np.array([8, 5, 4, 2, 3, 9])

print(arr)
index = np.where(arr >4)

print(index)
# Creating the desired array

new_arr = arr[index]

print(new_arr)
new_arr = np.delete(arr, index)
print(new_arr)
# Creating an image

img1 = np.zeros((20, 20))+3

img1[4:-4, 4:-4] = 6

img1[7:-7, 7:-7] = 9



plt.imshow(img1)

plt.show()
# filter out all values larger than 2 and less than 6.

index1 = img1 > 2

index2 = img1 > 6

compund_index = index1 & index2



img2 = np.copy(img1)



img2[compund_index] =0

plt.imshow(img2)

plt.show()
# The compound statement can alternatively be written as



compund_index = (img1 > 3) & (img1 < 7)

img3 = np.copy(img1)

img3[compund_index] = 0

plt.imshow(img3)

plt.show()
# Defining the matrices

A = np.matrix([[3, 6, -5],

              [1, -3, 2],

              [5, -1, 4]])



B = np.matrix([[12],

              [-2],

              [10]])



# Solving for the variables, where we invert A



X = A**(-1)*B

print(X)
#Answer check

3*X[0]+6*X[1]-5*X[2]
x = np.linalg.inv(A).dot(B)

print(x)