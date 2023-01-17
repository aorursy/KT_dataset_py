# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Example Python program for finding the min value along the given axis of an ndarray



 



# Import numpy module



import numpy as np



 



# Import standard library module random



import random



 



# Create a 3-Dimensional ndarray object



array_3d = np.array([[[1,1,1,1],



                     [2,2,2,2],



                     [3,3,3,3]],



                   



                     [[4,4,4,4],



                     [5,5,5,5],



                     [6,6,6,6]],



                   



                     [[7,7,7,7],



                     [8,8,8,8],



                     [9,9,9,9]]]



                    )



                   



# Print the 3-Dimensional Array



print("Input ndarray:")



print(array_3d)



 



print("Shape of the ndarray:")



print(array_3d.shape)



 



print("Number of dimensions/axis of the ndarray:")



print(array_3d.ndim)



print("Input ndarray:")



print(array_3d)





                   



# Print the minimum value of the whole array - without considering the axis parameter



print("Minimum value in the whole array:%d"%(array_3d.min()))                   



 



# Print the minimum value for axis = 0



print("Minimum value along the axis 0:")



print(array_3d.min(axis=0))



 



# Print the minimum value for axis = 1



print("Minimum value along the axis 1:")



print(array_3d.min(axis=1))



 



# Print the minimum value for axis = 2



print("Minimum value along the axis 2:")



print(array_3d.min(axis=2))



 



print("Input ndarray:")



print(array_3d)



# Print the maximum value of the whole array - without considering the axis parameter



print("Maximum value in the whole array:%d"%(array_3d.max()))                    



 



# Print the maximum value for axis = 0



print("Maximum value along the axis 0:")



print(array_3d.max(axis=0))



 



# Print the maximum value for axis = 1



print("Maximum value along the axis 1:")



print(array_3d.max(axis=1))



 



# Print the maximum value for axis = 2



print("Maximum value along the axis 2:")



print(array_3d.max(axis=2))



 
# Python Program illustrating 

# numpy.mean() method 

import numpy as np 



# 1D array 

arr = [20, 2, 7, 1, 34000] 

arr = np.array(arr)

print("arr : ", arr) 

print("mean of arr : ", np.mean(arr)) 



# Python Program illustrating 

# numpy.mean() method 

import numpy as np 



#axis = 0 means along the column and axis = 1 means working along the row.

# 2D array 

arr = [[14, 17, 12, 33, 44], 

        [15, 6, 27, 8, 19], 

        [23, 2, 54, 1, 4, ]] 



# mean of the flattened array 

print("\nmean of arr, axis = None : ", np.mean(arr)) 

# mean along the axis = 0 

print("\nmean of arr, axis = 0 : ", np.mean(arr, axis = 0)) 



# mean along the axis = 1 

print("\nmean of arr, axis = 1 : ", np.mean(arr, axis = 1)) 



# Python Program illustrating 

# numpy.median() method 



import numpy as np 



# 1D array 

arr = [20, 2, 1, 7, 340000000] 



print("arr : ", arr) 

print("median of arr : ", np.median(arr)) 



# numpy.median() method 

import numpy as np 

# 2D array 

arr = [[14, 17,12, 33, 44], 

       [15, 6, 27, 8, 19], 

       [23, 2, 54, 1, 4, ]] 



#axis = 0 means along the column and axis = 1 means working along the row.



# median of the flattened array 

print("\nmedian of arr, axis = None : ", np.median(arr)) 

# median along the axis = 0 

print("\nmedian of arr, axis = 0 : ", np.median(arr, axis = 0)) 



# median along the axis = 1 

print("\nmedian of arr, axis = 1 : ", np.median(arr, axis = 1)) 



# importing libraries 

import numpy as np 



# sort along the first axis 

a = np.array([[12, 15], [10, 1]]) 



print ("Original Array : \n", a)

arr1 = np.sort(a, axis = 0) # row level 

print ("Along first axis : \n", arr1)





# sort along the last axis 

arr2 = np.sort(a, axis = 1) # column level 

print ("\nAlong first axis : \n", arr2) 





arr1 = np.sort(a, axis = None)

print ("\nAlong none axis : \n", arr1) 

##Reverse sorting 



arr = np.arange(10,0, -1)

print(arr)

arr2 = np.sort(arr)

print(arr2)

print(arr2[::-1])# reversing the array using indexing
# Variance 



# Var (X)  = Sum [( Xi - Xmean)^2 ] /N -1 this is called sample variance 

# Var (X)  = Sum [( Xi - Xmean)^2 ] /N this is called population variance 



x = np.array([1, 2, 3, 40] )

print(np.var(x))





print(np.std(x))   # square root of variance

# In probability theory and statistics, covariance is a measure of the joint variability of two random

# varibles





# Cov(X, Y) = Sum [( Xi - Xmean)*(Yi - Ymean)]/ N- 1  - this is called sample covariance 

# Cov(X, Y) = Sum [( Xi - Xmean)*(Yi - Ymean)]/ N  - this is called population covariance



# Not very useful

# Python code to demonstrate the 

# use of numpy.cov 

import numpy as np 



x = np.array([1, 2, 3, 4] )



y = np.array([2, 2, 3, 10] )



# # find out covariance with respect columns 

# cov_mat = np.stack((x, y), axis = 0) # row wise stacking

# print(cov_mat)



print(np.cov(x,y)) 



# [[ cov (x,x)    cov(x,y) ],

#  [cov (y,x)   cov(y,y)]]

# Correlation 



# Corr(X, Y) = Cov(X,Y)/ std(X)*std(Y)  



# value is between 0 an 1 





a = np.array([1,2,3,4])

b = np.array([2,2,3,10])

print(np.corrcoef(a,b))