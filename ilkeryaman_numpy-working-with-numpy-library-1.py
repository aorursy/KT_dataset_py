import numpy as np # linear algebra
data_list = [1, 2, 3]

data_list2 = [[10, 20, 30], [40, 50, 60], [70, 80, 90]]

data_list3 = []
arr = np.array(data_list)

arr2 = np.array(data_list2)
arr
arr2
arr2[2, 2] # Getting inner item value from multi-dimensional array
np.arange(25) # Create an array, by putting values from 0 to 25
np.arange(10, 20) # Create an array, by putting values from 10 to 20
np.arange(0, 100, 5) # Create an array, by putting values from 0 to 100, 5 by 5.
np.linspace(0, 100, 5) # Create an array, by putting values from 0 to 100, by getting 5 values which has same distance.
np.linspace(1, 6, 6) # Create an array, by putting values from 1 to 6, by getting 6 values which has same distance.
np.zeros(5) # put 0 for 5 times
np.zeros((2, 3)) # Create a two-dimensional array which has 6 items. Put 0's in each item.
np.ones(6) # put 1 for 6 times
np.ones((2, 3)) # Create a two-dimensional array which has 6 items. Put 1's in each item.
np.eye(4) # Return a 2-D array with ones on the diagonal and zeros elsewhere.
np.random.randint(0, 10) # Generate a random integer between 0 and 10.
np.random.randint(10) # Another way to generate random interger between 0 and 10.
np.random.randint(1, 10, 5) # Generate 5 random number between 1 and 10.
np.random.randn(5) # Generate 5 random number from Gaussian distribution.
np.random.rand(10) # Attention! Generate 10 random double.
np.random.rand(1, 10) # Generate 1 array with 10 random double values. Put it in an array.
np.random.rand(3, 5) # Generate 3 arrays with 5 random double values. Put it in an array.
np.arange(25).reshape(5, 5) # Generate 25 values and separate them to 5-5 sized arrays.
new_array = np.random.randint(1, 100, 10) # Generate 10 numbers between 1 and 100.
new_array
new_array.max() # maximum value at new_array
new_array.min() # minimum value at new_array
new_array.sum() # sum of all values at new_array
new_array.mean() # average of values at new_array
new_array.argmax() # index of maximum value at new_array
new_array.argmin() # index of minimum value at new_array