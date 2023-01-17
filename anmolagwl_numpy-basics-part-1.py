# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np 

import os

#create one dimension NumPy array -
np.array([10,20,30,40,50])

np.array([10.1,20.2,30.3,40,50])
#create n dimension NumPy array -
np.array([[10, 20, 30], [40, 50, 60]])    # 2 x 3 array, means 2 rows and 3 columns

#Array filled with zeros -
np.zeros(3)
np.zeros(3, dtype = int)
#Similar to zeros we can create array of 1s -
np.ones(3, dtype = int)
#Change the above to multidimension -
np.ones((3,4), dtype = int)
#Apart from 0 and 1 we can also use other numbers -
np.full((3, 4), 100)
#One more example-
np.full(3, 200)
#eye function - to create identity matrix
np.eye(3)
np.eye(3, dtype = int)
#empty function - to create an uninitialized array. The values will be whatever happens to already exist at that memory location
np.empty(5)
#random function
#Tt has various functions. Let's see some of them -

np.random.random(5)  #Create a 1d array of random values between 0 and 1
#The above can also be done with rand -
np.random.rand(5)
#Similarly 2d and 3d random arrays can be created -
np.random.random((5,5)) # Note that it will take 2 parantheses
np.random.rand(5,5)
np.random.random((5,5,5))
np.random.rand(5,5,5)
#random.randint function -

np.random.randint(1,10,5) #This will create a 1d array of 5 random integers between 1 and 10 (10 is exclusive)
np.random.randint(1,10,(2,3)) #This will create array of 2x3
np.random.randint(1,10,(2,3,3)) #This will create a 3d array
#random.normal fucntion - It generates random numbers from the normal distribution -

np.random.normal(2,2,5).astype('int32') #This will create an array of 5 random numbers with mean 2 and standard deviation 2
np.random.normal(2,2,(5,5)).astype('int32')
#arange function
np.arange(10,100,2) #self explanatory

#linspace function 
np.linspace(10,20,4) #It create an array of 4 values evenly spaced between 10 and 20
#We can also specity dtype in linspace -
np.linspace(10,20,4,dtype=int)