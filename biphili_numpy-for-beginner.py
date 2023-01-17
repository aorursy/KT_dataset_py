# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
my_list1=[1,2,3,4]

my_list1
# Making an array using a list

my_array1=np.array(my_list1)

my_array1
my_list2=[11,22,33,44]
#Combining the two lists 

my_lists=[my_list1,my_list2]
#Creating a 2D array

my_array2=np.array(my_lists)

my_array2
my_array2.shape
#Finding out the datatype of the array 

my_array2.dtype
#Creating an array of all zeros

my_zeros_array=np.zeros(5)

my_zeros_array                   
#Checking the datatype of Zero Array 

my_zeros_array.dtype
#Creating an array of ones 

np.ones([5,5])
#Creating empty arrays 

np.empty(5)
#Creating an identity matrix

np.eye(5)
np.arange(5)
#Creating an array starting with 5 and enfing with 50 with a space of 2

np.arange(5,50,2)
arr1=np.array([[1,2,3,4],[8,9,10,11]])

arr1
#Multiplying two arrays 

arr1*arr1
#Subtraction on arrays

arr1-arr1
#Scalar multiplication 

1/arr1
#Exponenetial operation

arr1**3