# importing the core library

import numpy as np



import os

for dirname, _, filenames in os.walk('../input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

# print multiple output in single cell

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"


import numpy as np

print(np.__version__)
# Question : Create a 1D array of numbers from 0 to 9

# Output : #> array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])



# Solution

X = np.arange(10)

X
# Question : Create a 3×3 numpy array of all True’s



# Solution

np.full((3,3), True, dtype=bool)



#or

np.full((9), True, dtype=bool).reshape(3,3)



#or

np.ones((3,3), dtype=bool)



#or

np.ones((9), dtype=bool).reshape(3,3)
# Question : Extract all odd numbers from array

# input: arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

# output: array([1, 3, 5, 7, 9])



#Solution



arr = np.arange(10)



arr[arr%2 == 1]
# Question: Replace all odd numbers in arr with -1

# input: arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

# output: array([ 0, -1,  2, -1,  4, -1,  6, -1,  8, -1])



# Solution



arr = np.arange(10)



arr[arr%2 == 1] = -1

arr
# Question: Replace all odd numbers in arr with -1 without changing arr

# input: arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

# output: out

# array([ 0, -1,  2, -1,  4, -1,  6, -1,  8, -1])

# arr

# array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])



# Solution



arr = np.arange(10)



out = arr.copy()



out[out%2 == 1] = -1



print('Modified Array')

out



print('\nOriginal Array')

arr
# Question: Convert a 1D array to a 2D array with 2 rows

# input: np.arange(10)

# output array([[0, 1, 2, 3, 4],

#               [5, 6, 7, 8, 9]])



# Solution



arr = np.arange(10)

arr.reshape(2,5)



# Another solution

arr = np.arange(10)

arr.reshape(2, -1)  # Setting to -1 automatically decides the number of cols
# Question: Stack arrays a and b vertically

# input: a = np.arange(10).reshape(2,-1)

#        b = np.repeat(1, 10).reshape(2,-1)



# output: array([[0, 1, 2, 3, 4],

#                [5, 6, 7, 8, 9],

#                [1, 1, 1, 1, 1],

#                [1, 1, 1, 1, 1]])



# Solution



a = np.arange(10).reshape(2,-1)

b = np.repeat(1, 10).reshape(2,-1)



np.vstack([a,b])
# Question: Stack the arrays a and b horizontally.



# Input: a = np.arange(10).reshape(2,-1)

#        b = np.repeat(1, 10).reshape(2,-1)

# Output: array([[0, 1, 2, 3, 4, 1, 1, 1, 1, 1],

#                [5, 6, 7, 8, 9, 1, 1, 1, 1, 1]])





# Solution:

a = np.arange(10).reshape(2,-1)

b = np.repeat(1, 10).reshape(2,-1)



np.hstack([a,b])
# Question: Create the following pattern without hardcoding. Use only numpy functions and the below input array a.



# Input: a = np.array([1,2,3])

# Output: array([1, 1, 1, 2, 2, 2, 3, 3, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3])





# Solution



a = np.array([1,2,3])

np.r_[np.repeat(a, 3), np.tile(a, 3)]



#other solution

np.hstack((np.repeat(a, 3), np.tile(a, 3)))
# Question: Get the common items between a and b



# Input: a = np.array([1,2,3,2,3,4,3,4,5,6])

#        b = np.array([7,2,10,2,7,4,9,4,9,8])



# Output: array([2, 4])





# Solution:

a = np.array([1,2,3,2,3,4,3,4,5,6])

b = np.array([7,2,10,2,7,4,9,4,9,8])

np.intersect1d(a,b)
# Question: From array a remove all items present in array b



# Input: a = np.array([1,2,3,4,5])

#        b = np.array([5,6,7,8,9])



# Output: array([1,2,3,4])





# Solution

a = np.array([1,2,3,4,5])

b = np.array([5,6,7,8,9])



np.setdiff1d(a,b)
# Question: Get the positions where elements of a and b match



# Input: a = np.array([1,2,3,2,3,4,3,4,5,6])

#        b = np.array([7,2,10,2,7,4,9,4,9,8])



# Output: (array([1, 3, 5, 7]),)





# Solution



a = np.array([1,2,3,2,3,4,3,4,5,6])

b = np.array([7,2,10,2,7,4,9,4,9,8])



np.where(a == b)
# Question: Get all items between 5 and 10 from a.



# Input: a = np.array([2, 6, 1, 9, 10, 3, 27])

# Output: (array([6, 9, 10]),)





# Solution



a = np.array([2, 6, 1, 9, 10, 3, 27])

a[(a >= 5) & (a <= 10)]
# Question: Convert the function maxx that works on two scalars, to work on two arrays.

# Input:



def maxx(x, y):

    """Get the maximum of two items"""

    if x >= y:

        return x

    else:

        return y



# maxx(1, 5)

#> 5



# Output:

# a = np.array([5, 7, 9, 8, 6, 4, 5])

# b = np.array([6, 3, 4, 8, 9, 7, 1])

# pair_max(a, b)

# array([ 6.,  7.,  9.,  8.,  9.,  7.,  5.])



# Solution



def pair_max(x, y):

    # here I am using map to make tuple from a and b, other solution is using zip(a,b)

    maximum = [maxx(a,b) for a,b in map(lambda a,b:(a,b),x,y)]

    # using zip

    # maximum = [maxx(a,b) for a,b in zip(x,y)]

    return np.array(maximum)



a = np.array([5, 7, 9, 8, 6, 4, 5])

b = np.array([6, 3, 4, 8, 9, 7, 1])



pair_max(a,b)
# Question: Swap columns 1 and 2 in the array arr.



# Input:



arr = np.arange(9).reshape(3,3)



print('Original array')

arr



# Solution



print("\nModified array")

arr[:, [1,0,2]]
# Question: Swap rows 1 and 2 in the array arr:



# Input: 



arr = np.arange(9).reshape(3,3)

print('Original array')

arr



# Solution



print("\nModified array")

arr[[1,0,2], :]
# Question: Reverse the rows of a 2D array arr.



# Input:



arr = np.arange(9).reshape(3,3)



print('Original array')

arr



# Solution



print("\nModified array")

arr[::-1, :]
# Question: Reverse the columns of a 2D array arr.



# Input: arr = np.arange(9).reshape(3,3)



# Solution



arr = np.arange(9).reshape(3,3)

print('Original array')

arr





print("\nModified array")

arr[:, ::-1]
# Question: Create a 2D array of shape 5x3 to contain random decimal numbers between 5 and 10.



# Solution:



rand_arr = np.random.uniform(5,10, size=(5,3))

rand_arr
# Question: Print or show only 3 decimal places of the numpy array rand_arr.



# Input: rand_arr = np.random.random((5,3))



rand_arr = np.random.random((5,3))

np.set_printoptions(precision=3)

rand_arr
# Pretty print rand_arr by suppressing the scientific notation (like 1e10)



# Input: 

# Create the random array

np.random.seed(100)

rand_arr = np.random.random([3,3])/1e3

np.set_printoptions(suppress=False)

rand_arr



# Output:

#> array([[ 0.000543,  0.000278,  0.000425],

#>        [ 0.000845,  0.000005,  0.000122],

#>        [ 0.000671,  0.000826,  0.000137]])



np.set_printoptions(suppress=True)

rand_arr

#> array([[ 0.000543,  0.000278,  0.000425],

#>        [ 0.000845,  0.000005,  0.000122],

#>        [ 0.000671,  0.000826,  0.000137]])
# Question: Limit the number of items printed in python numpy array a to a maximum of 6 elements.

a = np.arange(15)

np.set_printoptions(threshold=6)

a
# Question: Print the full numpy array a without truncating.



# Input: np.set_printoptions(threshold=6)

# a = np.arange(15)

# a



# Output: a

#> array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14])



# Solution



a = np.arange(15)





np.set_printoptions(threshold=15)

a
# Question: Import the iris dataset keeping the text intact.



# Solution:

iris_data = np.genfromtxt('../input/iris/Iris.csv', delimiter=',', skip_header=1, 

                          usecols = [0,1,2,3,4,5], dtype = object)

iris_data
# Question: Extract the text column species from the 1D iris imported in previous question.



data = np.genfromtxt('../input/iris/Iris.csv', delimiter=',', skip_header=1, 

                          usecols = [-1], dtype = object)

data
# Question: Convert the 1D iris to 2D array iris_2d by omitting the species text field.

iris_data = np.genfromtxt('../input/iris/Iris.csv', delimiter=',', skip_header=1, dtype='float', usecols=[0,1,2,3])

iris_data
# Question: Find the mean, median, standard deviation of iris's sepallength (1st column)



iris_data = np.genfromtxt('../input/iris/Iris.csv', delimiter=',', skip_header=1, usecols = [1])



print('Mean', np.mean(iris_data))

print('Median', np.median(iris_data))

print('Standard Deviation', np.std(iris_data))
# Question: Create a normalized form of iris's sepallength whose values range exactly between 0 and 1 so that the minimum has value 0 and maximum has value 1.



# Solution



iris_data = np.genfromtxt('../input/iris/Iris.csv', delimiter=',', dtype='float', usecols=[1], skip_header=1)



(iris_data - np.min(iris_data))/(np.max(iris_data) - np.min(iris_data))
# Question: Compute the softmax score of sepallength.



iris_data = np.genfromtxt('../input/iris/Iris.csv', delimiter=',', dtype='float', usecols=[1], skip_header=1)

softmax = np.exp(iris_data)/sum(np.exp(iris_data))

softmax.sum() # it must sum 1

softmax
# Question. Find the 5th and 95th percentile of iris's sepallength

iris_data = np.genfromtxt('../input/iris/Iris.csv', delimiter=',', dtype='float', usecols=[1], skip_header=1)



np.percentile(iris_data, q=[5, 95])
# Question: Insert np.nan values at 20 random positions in iris_2d dataset



# Solution

iris_data = np.genfromtxt('../input/iris/Iris.csv', delimiter=',', dtype='float', usecols=[1,2,3,4], skip_header=1)

for i in np.random.randint(0, len(iris_data), 20):

    iris_data[i]=np.nan

iris_data
# Question: Find the number and position of missing values in iris_2d's sepallength (1st column)



iris_data = np.genfromtxt('../input/iris/Iris.csv', delimiter=',', dtype='float', usecols=[1,2,3,4], skip_header=1)

iris_data[np.random.randint(len(iris_data), size=20),np.random.randint(4,size=20)] = np.nan



# Find total mising value in complete data

print("Number of missing values in Iris data: \n", np.isnan(iris_data[:, :]).sum())



# Find total mising value in 1D data

print("Number of missing values in any one feature of Iris data: \n", np.isnan(iris_data[:, 0]).sum())



print("Position of missing values: \n", np.where(np.isnan(iris_data[:, 0])))
# Question: Filter the rows of iris_2d that has petallength (3rd column) > 1.5 and sepallength (1st column) < 5.0



iris_data = np.genfromtxt('../input/iris/Iris.csv', delimiter=',', dtype='float', usecols=[1,2,3,4], skip_header=1)



# Solution

iris_data[(iris_data[:, 2] > 1.5) & (iris_data[:, 0] < 5.0)]
# Question: Select the rows of iris_2d that does not have any nan value.



diabetes_data = np.genfromtxt('../input/pima-indians-diabetes-database/diabetes.csv', delimiter=',', dtype='float', usecols=[0,1,2,3,4,5,6,7], skip_header=1)

diabetes_data[np.random.randint(150, size=20), np.random.randint(4, size=20)] = np.nan

diabetes_data[np.sum(np.isnan(diabetes_data), axis = 1) == 0][:5]
# question: Find the correlation between SepalLength(1st column) and PetalLength(3rd column) in iris_2d

# insted or using iris data I am going to used pima diabetes data and going to find corelation between BP(1st column) and BMI (5th column).



diabetes_data = np.genfromtxt('../input/pima-indians-diabetes-database/diabetes.csv',

                              delimiter=',', dtype='float', usecols=[0,1,2,3,4,5,6,7], skip_header=1)



print(np.corrcoef(diabetes_data[:, 1], diabetes_data[:, 5]))



print('\n')

# you can get correlation by getting value at index [0,1] or [1,0]

print(np.corrcoef(diabetes_data[:, 1], diabetes_data[:, 5])[0,1])
# question: Find out if iris_2d has any missing values.

diabetes_data = np.genfromtxt('../input/pima-indians-diabetes-database/diabetes.csv',

                              delimiter=',', dtype='float', usecols=[0,1,2,3,4,5,6,7], skip_header=1)



np.isnan(diabetes_data).any()
# Question: Replace all ccurrences of nan with 0 in numpy array



wine_quality = np.genfromtxt('../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv',

                             delimiter=',', dtype='float', usecols=[0,1,2,3,4,5,6,7,8,9,10], skip_header=1)



wine_quality[np.random.randint(len(wine_quality), size=20), np.random.randint(11, size=20)] = np.nan



print("Does dataset have any Nan value:",np.isnan(wine_quality).any())



wine_quality[np.isnan(wine_quality)] = 0



print("Does dataset have any Nan value:",np.isnan(wine_quality).any())
# Question: Find the unique values and the count of unique values in mashroom data's habitat (22 column) column



mushroom = np.genfromtxt('../input/mushroom-classification/mushrooms.csv',

                             delimiter=',', dtype=object, usecols=[0], skip_header=1)

#mushroom

np.unique(mushroom, return_counts=True)
# Question: Bin the petal length (3rd) column of iris_2d to form a text array, such that if petal length is:



# Less than 3 --> 'small'

# 3-5 --> 'medium'

# >=5 --> 'large'



# Solution



iris_data = np.genfromtxt('../input/iris/Iris.csv', delimiter=',', dtype=object, usecols=[3], skip_header=1)



bins = np.array([0, 3, 5, 7])

inds = np.digitize(iris_data.astype('float'), bins)



labels = {1:'small', 2: 'medium', 3:'large'}

iris_cat_data = [labels[x] for x in inds]

iris_cat_data[:10]
# Question: Create a new column for volume in iris_2d, where volume is (pi x petallength x sepal_length^2)/3



iris_data = np.genfromtxt('../input/iris/Iris.csv', delimiter=',', 

                          dtype=object, usecols=[1,2,3,4], skip_header=1)



sepallength = iris_data[:, 0].astype('float')

petallength = iris_data[:, 2].astype('float')



new_column = (np.pi * petallength * (sepallength**2))/3



new_column = new_column[:, np.newaxis]

#new_column



# Add the new column

out = np.hstack([iris_data, new_column])



# View

out[:4]
# Question: Randomly sample iris's species such that setose is twice the number of versicolor and virginica



iris_data = np.genfromtxt('../input/iris/Iris.csv', delimiter=',', 

                          dtype=object, usecols=[1,2,3,4,5], skip_header=1)



np.random.seed(100)

a = np.array(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])

species_out = np.random.choice(a, 150, p=[0.5, 0.25, 0.25])

print(np.unique(species_out, return_counts=True))
# Question: What is the value of second longest petallength of species setosa

# For this question I am going to find second highest bloodpressure (2nd column) where outcome is 1

diabetes_data = np.genfromtxt('../input/pima-indians-diabetes-database/diabetes.csv',

                              delimiter=',', dtype=object, usecols=[0,1,2,3,4,5,6,7,8], skip_header=1)



# Solution

bloodpressure= diabetes_data[diabetes_data[:, 8]==b'1', [2]].astype('float')



np.unique(np.sort(bloodpressure))[-2]
# Question: Sort the iris dataset based on sepallength column.

# In this problem, I am going to sort the diabetes dataset based on Glucose (1th column)

diabetes_data = np.genfromtxt('../input/pima-indians-diabetes-database/diabetes.csv',

                              delimiter=',', dtype=object, usecols=[0,1,2,3,4,5,6,7,8], skip_header=1)



diabetes_data[diabetes_data[:,1].argsort()]
# Question: Find the most frequent value of petal length (3rd column) in iris dataset.



iris_data = np.genfromtxt('../input/iris/Iris.csv', delimiter=',', 

                          dtype=object, usecols=[1,2,3,4,5], skip_header=1)



v,c = np.unique(iris_data[:, 2], return_counts=True)

v[np.argmax(c)]
# Question: Find the position of the first occurrence of a value greater than 1.0 in 

# petalwidth 4th column of iris dataset.

iris_data = np.genfromtxt('../input/iris/Iris.csv', delimiter=',', 

                          dtype=object, usecols=[4], skip_header=1)



np.argwhere(iris_data[:].astype(float) > 1.0)[0]
# Question: From the array a, replace all values greater than 30 to 30 and less than 10 to 10.



# Solution



np.set_printoptions(precision=2)

np.random.seed(100)

a = np.random.uniform(1,50, 20)



a[a<10]=10

a[a>30]=30

np.set_printoptions(threshold=20)

a
# Question: Get the positions of top 5 maximum values in a given array a.

np.random.seed(100)

a = np.random.uniform(1,50, 20)

a

sort = a.argsort()

print('Positions')

sort[-5:][::-1]

print('Values')

a[sort][-5:][::-1]
# Question: Compute the counts of unique values row-wise.



# Solution

def counts_of_all_values_rowwise(arr2d):

    # Unique values and its counts row wise

    num_counts_array = [np.unique(row, return_counts=True) for row in arr2d]



    # Counts of all values row wise

    return([[int(b[a==i]) if i in a else 0 for i in np.unique(arr2d)] for a, b in num_counts_array])



np.random.seed(100)

np.set_printoptions(threshold=10)

arr = np.random.randint(1,11,size=(6, 10))

arr

print(np.arange(1,11))

counts_of_all_values_rowwise(arr)
# Question: Convert array_of_arrays into a flat linear 1d array.

arr1 = np.arange(3)

arr2 = np.arange(3,7)

arr3 = np.arange(7,10)



arr_2d = np.concatenate([arr1, arr2, arr3])

print(arr_2d)