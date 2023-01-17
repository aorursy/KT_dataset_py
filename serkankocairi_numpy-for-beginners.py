# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



 # linear algebra





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#we import the library and np = numpy

import numpy as np
list1 = [1,2,3,4,5,6,7,8,9,10]

print(list1)



array1 = np.array(list1) #np.array() converts to an array

array1
np.arange(1,15) #np.arange()  creates an array from 1 to 14.
np.arange(1,15,2) #the same but each progresses by jumping 2
np.linspace(-1,1,15) #np.linspace() creates 15 equal numbers between -1 and 1.
np.logspace(0,2,3) #np.logspace() creates [0, 1, 2] => [10^0, 10^1, 10^2]
#np.zeros() function enables we to create  arrays that contain only zeros 

np.zeros(100) 
#np.ones() function enables we to create  arrays that contain only ones 

np.ones(50) 
#np.zeros(()) -> if we want to create a matrix we should use double brackets.

#25 rows and 4 columns

np.zeros((25,4)) 
np.eye(5) #np.eye() creates a identity matrix. 5 rows and 5 columns.
np.random.randint(0,50) #np.random.randint() creates a random number between 0 and 50
np.random.randint(0,50,5) #creates 5 random numbers betwenn 0 and 50
np.random.rand(3) #np.random.rand() creates 3 random numbers betwenn 0 and 1
data1 = np.arange(10) #we created an array

data1
#The reshape() function is used to give a new shape to an array without changing its data

data1.reshape(2,5)
data2 = np.array([1,0,5,4,6,3,7,8,2,10]) #we created a new array

data2
#cumsum() function is used when we want to compute the cumulative sum of array elements over a given axis.

#      ex = [0,1,2,3]

#ex.cum() = [0,0+1,0+1+2,0+1+2+3] = [0,1,3,6]

data2.cumsum() 
data2.min() #min() is finds the smallest value in the array
data2.max() #max() is finds the largest value in the array
data2.sum() #sum() is  adds all values in the array
#argmax() is finds the index of the largest value in the array

data2.argmax() 
#argmin() is finds the index of the smallest value in the array

data2.argmin()
data3 = np.array([[8,3],[2,9]]) #we created a new matrix. 2 rows and 2 columns

data3
np.linalg.det(data3) #np.ligalg.det() is finds determinant.
np.std(data3) #np.std() is finds standard deviation
np.var(data3) #np.var is finds variance. variance = standard deviation^2
data1 #Let's do some operations for data1
data1[:5] #finds the first 5 elements of the array.
data1[::2] #Find two by two from beginning to end
data1[:3]=8 #finds the first 3 elements and them change is with 8

data1
data1[::-1] #reverse the array
matrix1 = np.array([[5,6],[7,4]])

matrix2 = np.array([[0,3],[9,2]])



      
#if axis=0, there are 4 rows and 2 columns in matrix

matrix_0=np.concatenate([matrix1,matrix2],axis=0)

matrix_0
#if axis=1, there are 2 rows and 4 columns in matrix

matrix_1=np.concatenate([matrix1,matrix2],axis=1)

matrix_1
matrix_0.shape #shape is shows the number of rows and columns
