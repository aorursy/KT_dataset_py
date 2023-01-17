# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#One dimentional array

a = np.array([5,6,7])

print(a)

print(a[1,])

a.ndim

a.shape
#two dimentional array

b = np.array([[10,11,12], [100,10,20]])

print(b)

b.ndim

b.shape

b.size
#access element in a 2d array

big_array = np.array([[1,2,3,4,5,6,7,8],

                     [9,10,11,12,13,14,15,16]])

big_array.shape

print(big_array[1,-2])



#access 2nd row

print(big_array[1,:])



#access 3rd column

print(big_array[:,2])



#access 3,5,7th element in 2nd row

print(big_array[1,2:8:2])



#assign/replace specific element

big_array[0,4] = 100

print(big_array)



#replace a whole column

big_array[:, 7] = [20,21]



print(big_array)



#replace an entire row

big_array[1,:] = range(500, 540, 5)



print(big_array)
#access elements in a 3d array

big_3d_array = np.array([[

                        [5,7,8],

                        [50,70,80]

                        ],

                        

                        [

                        [10,15,20],

                        [100,150,200]

                        ]])

big_3d_array.shape

big_3d_array.ndim
#initialize array with zeros

arr_zeros = np.zeros((2,3), dtype='int32')

print (arr_zeros)



#initialize array with ones

arr_ones = np.ones((3,2), dtype='int32')

print(arr_ones)



#intialzie array with any other number

arr_my_choice = np.full((4,3), 5.0)

print(arr_my_choice)



#initialize array with random numbers

arr_rand = np.random.rand(4,5,3)

print(arr_rand)
#Mathematics with numpy

math_arr = np.array([4,5,6])

print (math_arr)



# +

math_arr += 10

print (math_arr)



# -

math_arr = math_arr * 3

print(math_arr)



# cosine

print(np.cos(math_arr))
#matrix multiplication

mat_one = np.array([

    [10,10,1],

    [90,1,90]

])



print (mat_one)



mat_two = np.ones((3,2))

print (mat_two)





mat_multiplied = np.matmul(mat_one, mat_two)



print (mat_multiplied)
#statistics with numpy

stats_arr = np.array([

    [10,2,13],

    [-1,-100,-900]

])



print(np.min(stats_arr))

print(np.min(stats_arr, axis=0))

print(np.min(stats_arr, axis=1))

#reshape or reorganize arrays

original_array = np.array([[1,2,3,4,5,6,7,8]])

original_array.shape



new_array = original_array.reshape(2,4)

print(new_array)
#advanced array index operations

test_arr = np.array([

    [10,12,100,11,150],

    [1,500,400,2,-1]

])



test_arr > 100



test_arr[[0,1], [1]]