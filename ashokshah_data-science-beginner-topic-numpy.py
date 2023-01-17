#import Numpy library
import numpy as np
#Creating NumPy Arrays from List
my_list=[1,2,4,8]
my_list
#Creat arrays from the matrix
my_matrix = [[1,2,5],[8,9,6],[9,6,3]]
my_matrix
np.array(my_matrix)
#arange (You can generate a series from inital to final, intial included only)
np.arange(0,10)
#Zeros and Ones (You can generate arrays of zeros and ones)
np.zeros(5)
np.ones(3)
np.ones((3,3))
np.zeros ((2,2))
#Linspace can be used to generate evenly spaced numbers over a particular interval

np.linspace(0,15,4)
np.linspace(0,15,10)
np.linspace(0,10,50)
#Eye used to create an Identity Matrix

np.eye(5)
#Rand from Random library which generates random number arryas which is uniform
np.random.rand(6)
#You can either import random libray but for now I will be adding random library in the code
np.random.rand(3,3)
#Randn returns a sample or samples form the "Standard normal" distribution.
np.random.randn(5)
#Randint returns random integer from low (inclusive) to high (exclusive)
np.random.randint(1,50)
#If you want a series of random integer you can add interval in the syntax.
np.random.randint(1,50,20)
#"Attributes and Methods of Array"
#Gives you sorted array
np.arange(25)
#Reshape returns an array cotaining the same data with a new shape
my_rand=np.random.randint(0,50,9)
my_rand
my_rand.reshape(3,3)
my_rand2=np.arange(25)
my_rand2
my_rand2.reshape(5,5)
# "Different aggregates you can use on the arrays for computation purpose or finding the index location"
#max
my_rand2.max()
#min
my_rand.min()
#argmax (It returns the position of the max value in the array)
my_rand2.argmax()
#argmin (It returns the position of the min value in the array)
my_rand2.argmin()
my_rand3=np.random.randint(0,15,10)
my_rand3
#Shape gives out the shape of the array and it is an attribute that an arrays has. It is not a method
my_rand3.shape
#Now lets reshape the array to something else and then again check its shape
my_rand4=my_rand3.reshape(1,10)
#It has changed to 2D array.
my_rand4
my_rand4.shape
#dtype returns the data type of the array
my_rand4.dtype
arr1=np.arange(0,10)
arr2=np.arange(0,10)
# + to add arrays
arr1+arr2
# - to subtract arrays
arr1-arr2
# Double * means you multiplying it twice. So * can be used to get squares,cubes, roots
arr3=arr1**2
arr3
#sqrt to get square roots
np.sqrt(arr3)
arr1
# [] use to get a value at particular Index
arr1[5]
# [:] use to get values in a range where final is exclusive
arr1[1:5]
# [:]= use to set values for a range of index
arr1[0:4]=20
arr1
arr2 = np.arange(0,10)
arr2
#Important thing to note here, lets say
part_of_arr1 = arr1[0:4]
part_of_arr1
part_of_arr1[:]=2
part_of_arr1
#Lets check arr1
arr1
#In NumPy the data is not copied but rather it is the view of the original array. Thta can cause many problems.
#So to avoid it always get a copy of the original array.
arr_copied=arr1.copy()
arr_copied
#Now lets see 2D indexing
arr_2d = np.array(([11,27,35],[44,59,62],[73,80,99]))
arr_2d
#The indexing for 2D array has a format that is 
# Either array[row][col] or array[row,col]
arr_2d[1]
arr_2d[1][2]
arr_2d[1,2]
#Slicing a 2d array
arr_2d[:2,1:]
arr_2d[2,:]
arr_2d
#Comparison Operators
# >
arr_2d >45