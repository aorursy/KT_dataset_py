#1 A

import numpy as np



bool_arr = np.array([1, 0.5, 0, None, 'a', '', True, False], dtype=bool)

print(bool_arr)

#1 B

import numpy as np



#create numpy array

a = np.array([5, 8, 12])

print(a)
#1 C

my_list = [1, 2, 3, 4]             # Define a list



second_list = [5, 6, 7, 8]



two_d_array = np.array([my_list, second_list])



print(two_d_array)
#2

a=np.array([1,2,3,4,5,6])

answer = (a[a%2==1])

print (answer)
#3

import numpy as np

x = np.array([[ 0.42436315, 0.48558583, 0.32924763], [ 0.7439979,0.58220701,0.38213418], [ 0.5097581,0.34528799,0.1563123 ]])

print("Original array:")

print(x)

print("Replace all elements of the said array with .5 which are greater than .5")

x[x > .5] = .5

print(x)
#4

import numpy as np

array1 = np.array([0, 10, 20, 40, 60])

print("Array1: ",array1)

array2 = [10, 30, 40]

print("Array2: ",array2)

print("Common values between two arrays:")

print(np.intersect1d(array1, array2))
#6

import numpy



a = numpy.array([0, 1, 2, 3, 4, 5, 6])

b = numpy.array([6, 5, 4, 3, 2, 1, 6])

numpy.where(a==b)



    