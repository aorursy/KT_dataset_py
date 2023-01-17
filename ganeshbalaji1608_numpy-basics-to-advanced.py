# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra





#This Session is full of Numpy From Basics to Advance 
arr = np.array([1,2,3,4]) #single dimensional numpy array
arr[1] #to access the data
#Twodimensional array

arr2 = np.array([

    [1,2,3],

    [4,5,6]

])
#Multidimensional array

arr3 = np.array([

    [[1,2,3]],

    [[4,5,6]]

])
np.eye(3)
#Basic functions in python

print("shapes of the arrays, arr :{}, arr2 : {}, arr3 : {}".format( arr.shape, arr2.shape, arr3.shape))

print()

print("dimensional of the arrays, arr :{}, arr2 : {}, arr3 : {}".format( arr.ndim, arr2.ndim, arr3.ndim ))

print()

print("size of the arrays, arr :{}, arr2 : {}, arr3 : {}".format( arr.size, arr2.size,arr3.size))
#to Find Max

print("Max number")

print(arr.max())

print(arr2.max())

print(arr3.max())

print("Index number of the maximum arguments")

#to Find the max number argument

print(arr.argmax())

print(arr2.argmax())

print(arr3.argmax())
np.arange(1,100,2)
np.linspace(0,1,10)
np.linspace(1,10) #if range not given, it takes default value 50
np.logspace(1,10,4)
np.ones(5)
np.zeros(5)
np.random.rand(4) #Generates random number between 0 to 1 in the range of specified number
np.random.randn(10) #Generates random number between Normal distributions in the range of specified number
np.random.randint(1,10,4)
np.random.random_integers(4) #Gives the random number between 1 to specified number
np.random.random_integers(4, size = (2,2))