# Allow several prints in one cell

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"



# importing the core library

import numpy as np



# helper functions to list the datasets available

def print_files():

    import os

    for dirname, _, filenames in os.walk('/kaggle/input'):

        for filename in filenames:

            print(os.path.join(dirname, filename))

print_files()
# Q. Import numpy as np and print the version number.



# Solution

print("Solution")

import numpy as np

print(np.__version__)
# Q. Create a 1D array of numbers from 0 to 9



# Desired Output

# > array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])



# Solution

print("Solution")

arr = np.arange(10)

arr
# Q. Create a 3×3 numpy array of all True’s



# Solution

print("Solution")

arr = np.repeat(True, 9).reshape(3, -1)

arr
# Q. Extract all odd numbers from arr

# Input 

arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

print("Input")

arr



# Desired Output

# > array([1, 3, 5, 7, 9])



# Solution

print("Solution")

odds = arr[arr%2 != 0]

odds
# Q. Replace all odd numbers in arr with -1

# Input 

arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

print("Input")

arr



# Desired Output

# >  array([ 0, -1,  2, -1,  4, -1,  6, -1,  8, -1])



# Solution

print("Solution")

arr[arr%2 != 0] = -1

arr
# Q. Replace all odd numbers in arr with -1 without changing arr

# Input 

arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

print("Input")

arr



# Desired Output

# out #>  array([ 0, -1,  2, -1,  4, -1,  6, -1,  8, -1])

# arr #>  array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])



# Solution

out = arr.copy()

out[out%2 != 0] = -1

print("Solution")

print("Modified array")

out

print("Original array")

arr
# Q. Convert a 1D array to a 2D array with 2 rows

# Input 

arr = np.arange(10)

print("Input")

arr



# Desired Output

# > array([[0, 1, 2, 3, 4],

# >        [5, 6, 7, 8, 9]])



# Solution

print("Solution: reshaped array")

arr.reshape(2, -1)
# Q. Stack arrays a and b vertically

# Input

print("Input")

a = np.arange(10).reshape(2,-1)

b = np.repeat(1, 10).reshape(2,-1)

a

b



# Desired Output

#> array([[0, 1, 2, 3, 4],

#>        [5, 6, 7, 8, 9],

#>        [1, 1, 1, 1, 1],

#>        [1, 1, 1, 1, 1]])



# Solution

print("Solution: verticaly stacked arrays")

np.vstack((a, b))
# Q. Stack the arrays a and b horizontally.

# Input

print("Input")

a = np.arange(10).reshape(2,-1)

b = np.repeat(1, 10).reshape(2,-1)



# Desired Output

# > array([[0, 1, 2, 3, 4, 1, 1, 1, 1, 1],

# >        [5, 6, 7, 8, 9, 1, 1, 1, 1, 1]])



# Solution

print("Solution: horizontally stacked arrays")

np.hstack((a, b))
# Q. Create the following pattern without hardcoding. Use only numpy functions and the below input array a.



# Input

print("Input")

a = np.array([1,2,3])

a



# Desired Output

# > array([1, 1, 1, 2, 2, 2, 3, 3, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3])



# Solution

print("Solution")

solution_array = np.hstack((np.repeat(a, 3), a, a, a)) # using repeat to generate the 111222.. sequence and hstack 3 times the original array

np.set_printoptions(threshold=len(solution_array)) # just to help us see all the array-

solution_array
# Q. Get the common items between a and b



# Input

print("Input")

a = np.array([1,2,3,2,3,4,3,4,5,6])

b = np.array([7,2,10,2,7,4,9,4,9,8])

a

b



# Desired Output

# array([2, 4])



# Solution

print("Solution")

np.unique(a[a == b])
# Q. From array a remove all items present in array b



# Input

print("Input")

a = np.array([1,2,3,4,5])

b = np.array([5,6,7,8,9])

a

b



# Desired Output

# array([1,2,3,4])



# Solution

print("Solution")

a[~np.isin(a,b)] # np.isin to find the common elements (returns an array of Booleans). To filter only False, use ~ (CTRL + ALT + 4)
# Q. Get the positions where elements of a and b match



# Input

print("Input")

a = np.array([1,2,3,2,3,4,3,4,5,6])

b = np.array([7,2,10,2,7,4,9,4,9,8])

a

b



# Desired Output

# > (array([1, 3, 5, 7]),)



# Solution

print("Solution")

np.where(a == b) # Notice: the solution if the INDEX but the values
# Q. Get all items between 5 and 10 from a.



# Input

print("Input")

a = np.array([2, 6, 1, 9, 10, 3, 27])

a



# Desired Output

# (array([6, 9, 10]),)



# Solution

print("Solution")

a[(a > 5) & (a < 10)]
# Q. Convert the function maxx that works on two scalars, to work on two arrays.



# Input



def maxx(x, y):

    """

    Get the maximum of two items

    """

    

    if x >= y:

        return x

    else:

        return y

print("Result of the maxx function")

maxx(1, 5)



print("Input")

a = np.array([5, 7, 9, 8, 6, 4, 5])

b = np.array([6, 3, 4, 8, 9, 7, 1])

a

b



# Desired Output

# pair_max(a, b)

#> array([ 6.,  7.,  9.,  8.,  9.,  7.,  5.])



# Solution

print("Solution")



def pair_max(a, b):

    return np.array([max(x, y) for x, y in zip(a, b)]) # using a list comprehension to find the max between 2 arrays elementwise (using zip) and convert to numpy array



pair_max(a, b)
# Q. Swap columns 1 and 2 in the array arr.



# Input

print("Input")

arr = np.arange(9).reshape(3,3)

arr



# Solution

print("Solution")

temp = arr[:,0].copy() # temporary variable

arr[:,0], arr[:,1] = arr[:,1], temp

arr
# Q. Swap rows 1 and 2 in the array arr:



# Input

print("Input")

arr = np.arange(9).reshape(3,3)

arr



# Solution

print("Solution")

temp = arr[0,:].copy() # temporary variable

arr[0,:], arr[1,:] = arr[1,:], temp

arr

# Q. Reverse the rows of a 2D array arr.



# Input

print("Input")

arr = np.arange(9).reshape(3,3)

arr



# Solution

print("Solution")

arr[::-1]

arr[::-1, :] # exactly the same
# Q. Reverse the columns of a 2D array arr.



# Input

print("Input")

arr = np.arange(9).reshape(3,3)

arr



# Solution

print("Solution")

arr[:, ::-1]
# Q. Create a 2D array of shape 5x3 to contain random decimal numbers between 5 and 10.



# Solution

print("Solution")

# randint first argument: lower bound, second argument: higher bound (if you put 10, it will max return 9, so we add 1), third argument: number of samples. Then we reshape and done.

np.random.randint(5, 11, 15).reshape(5, 3) 
# Q. Print or show only 3 decimal places of the numpy array rand_arr.



# Setting print options to default

np.set_printoptions(edgeitems=3,infstr='inf', linewidth=75, nanstr='nan', precision=8, suppress=False, threshold=1000, formatter=None)



# Input

print("Input")

rand_arr = np.random.random((5,3))

rand_arr



# Solution

print("Solution")

np.set_printoptions(precision=3)

rand_arr
# Q. Pretty print rand_arr by suppressing the scientific notation (like 1e10)



# Setting print options to default

np.set_printoptions(edgeitems=3,infstr='inf', linewidth=75, nanstr='nan', precision=8, suppress=False, threshold=1000, formatter=None)



# Input

print("Input")

np.random.seed(100)

rand_arr = np.random.random([3,3])/1e3

rand_arr



# Desired Output

# > array([[ 0.000543,  0.000278,  0.000425],

# >        [ 0.000845,  0.000005,  0.000122],

# >        [ 0.000671,  0.000826,  0.000137]])



# Solution

print("Solution")

np.set_printoptions(suppress=True)

rand_arr
# Q. Limit the number of items printed in python numpy array a to a maximum of 6 elements.



# Setting print options to default

np.set_printoptions(edgeitems=3,infstr='inf', linewidth=75, nanstr='nan', precision=8, suppress=False, threshold=1000, formatter=None)



# Input

print("Input")

a = np.arange(15)

a



# Desired Output

# > array([ 0,  1,  2, ..., 12, 13, 14])



# Solution

print("Solution")

np.set_printoptions(threshold=6)

a
# Q. Print the full numpy array a without truncating.



# Input

print("Input")

np.set_printoptions(threshold=6)

a = np.arange(15)

a



# Desired Output

# > array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14])



# Solution

print("Solution")

np.set_printoptions(threshold=len(a))

a
# Q. Import the iris dataset keeping the text intact.



# Input

# Use the iris dataset provided

print_files()



# All the available options of the numpy genfromtxt function

# numpy.genfromtxt(fname, dtype=<class 'float'>, comments='#', delimiter=None, skip_header=0, 

#                    skip_footer=0, converters=None, missing_values=None, filling_values=None, 

#                    usecols=None, names=None, excludelist=None, deletechars=" !#$%&'()*+, -./:;<=>?@[\]^{|}~", 

#                    replace_space='_', autostrip=False, case_sensitive=True, defaultfmt='f%i', 

#                    unpack=None, usemask=False, loose=True, invalid_raise=True, max_rows=None, encoding='bytes')[source]¶



# Solution

print("Solution")

iris = np.genfromtxt('/kaggle/input/iris/Iris.csv', delimiter=',', skip_header=1, usecols = [0, 1, 2, 3, 4, 5], dtype = None)

iris
# Q. Extract the text column species from the 1D iris imported in previous question.



# Input

# Use the iris dataset provided

print_files()



# Use this if you are working on your local machine

# url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

# iris_1d = np.genfromtxt(url, delimiter=',', dtype=None)



# Solution

print("Solution")

iris = np.genfromtxt('/kaggle/input/iris/Iris.csv', delimiter=',', skip_header=1, usecols = [5], dtype='str')

iris



# Solution from the website

print("Solution from website")

iris_1d = np.genfromtxt('/kaggle/input/iris/Iris.csv', delimiter=',', dtype=None)

species = np.array([row[5] for row in iris_1d])

species[:5]
# Q. Convert the 1D iris to 2D array iris_2d by omitting the species text field.



# Input

# Use the iris dataset provided

print_files()



# Use this if you are working on your local machine

# url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

# iris_1d = np.genfromtxt(url, delimiter=',', dtype=None)



# Solution

print("Solution")

iris = np.genfromtxt('/kaggle/input/iris/Iris.csv', delimiter=',', skip_header=1, usecols = [0, 1, 2, 3, 4], dtype = None)

iris[:4]



# Another solution from the website

print("Another solution from website")

iris_1d = np.genfromtxt('/kaggle/input/iris/Iris.csv', delimiter=',', dtype=None)

iris_2d = np.array([row.tolist()[:4] for row in iris_1d])

iris_2d[:4]
# Q. Find the mean, median, standard deviation of iris's sepallength (1st column)



# Input

# Use the iris dataset provided

print_files()



# Use this if you are working on your local machine

# url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

# iris_1d = np.genfromtxt(url, delimiter=',', dtype=None)



# Solution

print("Solution")

iris = np.genfromtxt('/kaggle/input/iris/Iris.csv', delimiter=',', skip_header=1, usecols = [1])

iris



import pandas as pd

pd.Series(iris).describe()



from scipy import stats 

stats.describe(iris) 



# Solution from the website

print("Another solution from the website")

mu, med, sd = np.mean(iris), np.median(iris), np.std(iris)

print(mu, med, sd)
# Q. Create a normalized form of iris's sepallength whose values range exactly between 0 and 1 so that the minimum has value 0 and maximum has value 1.



# Input

# Use the iris dataset provided

print_files()



# Use this if you are working on your local machine

# url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

# iris_1d = np.genfromtxt(url, delimiter=',', dtype=None)



# Solution

print("Solution")

iris = np.genfromtxt('/kaggle/input/iris/Iris.csv', delimiter=',', skip_header=1, usecols = [1])

(iris - np.min(iris))/(np.max(iris) - np.min(iris))



# Another solution from the website

print("Another solution from the website")

iris.ptp() # peak to peak. Basically the same as (np.max(iris) - np.min(iris))

(iris - np.min(iris))/iris.ptp()
# Q. Compute the softmax score of sepallength.



# Input

# Use the iris dataset provided

print_files()



# Use this if you are working on your local machine

# url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

# iris_1d = np.genfromtxt(url, delimiter=',', dtype=None)



# Solution

# The Softmax regression is a form of logistic regression that normalizes an input value into a vector of values that follows a probability distribution whose total sums up to 1. 

print("Solution")

iris = np.genfromtxt('/kaggle/input/iris/Iris.csv', delimiter=',', skip_header=1, usecols = [1])

softmax = np.exp(iris)/sum(np.exp(iris))

softmax.sum() # it must sum 1



# We can also apply this to more than 1 column.

iris = np.genfromtxt('/kaggle/input/iris/Iris.csv', delimiter=',', skip_header=1, usecols = [1, 2, 3, 4])

softmax = np.exp(iris)/sum(np.exp(iris))

softmax.sum() # We have 4 since we have 4 columns, each sums 1

softmax



# Solution from the website

print("Solution from the website")



def softmax(x):

    """Compute softmax values for each sets of scores in x.

    https://stackoverflow.com/questions/34968722/how-to-implement-the-softmax-function-in-python"""

    e_x = np.exp(x - np.max(x))

    return e_x / e_x.sum(axis=0)



print(softmax(iris).sum())

print(softmax(iris))

# Q. Find the 5th and 95th percentile of iris's sepallength



# Input

# Use the iris dataset provided

print_files()



# Use this if you are working on your local machine

# url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

# iris_1d = np.genfromtxt(url, delimiter=',', dtype=None)



# Solution

print("Solution")

iris = np.genfromtxt('/kaggle/input/iris/Iris.csv', delimiter=',', skip_header=1, usecols = [1])

iris

np.percentile(iris, q = [5, 95])
# Q. Insert np.nan values at 20 random positions in iris_2d dataset



# Input

# Use the iris dataset provided

print_files()



# Use this if you are working on your local machine

# url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

# iris_1d = np.genfromtxt(url, delimiter=',', dtype=None)



# Solution

print("Solution")

iris = np.genfromtxt('/kaggle/input/iris/Iris.csv', delimiter=',', skip_header=1, usecols = [1, 2, 3, 4])

index = np.random.randint(0, 150, 20)

iris[index] = np.nan

iris



# Solution from the website

print("Another solution from the website")

iris = np.genfromtxt('/kaggle/input/iris/Iris.csv', delimiter=',', skip_header=1, usecols = [1, 2, 3, 4])

i, j = np.where(iris_2d) # get the index of all 600 elements of the array

nan_index = [np.random.choice((i), 20), np.random.choice((j), 20)] # get some random values for each row and column

iris[nan_index] = np.nan

iris



# Solution 3 from the website

print("Solution 3 from the website")

iris = np.genfromtxt('/kaggle/input/iris/Iris.csv', delimiter=',', skip_header=1, usecols = [1, 2, 3, 4])

iris[np.random.randint(149, size=20), np.random.randint(4, size=20)] = np.nan

iris
#### Q. Find the number and position of missing values in iris_2d's sepallength (1st column)



# Input

# Use the iris dataset provided

print_files()



# Use this if you are working on your local machine

# url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

# iris_2d = np.genfromtxt(url, delimiter=',', dtype='float')

# iris_2d[np.random.randint(150, size=20), np.random.randint(4, size=20)] = np.nan



# Solution

print("Solution")

iris = np.genfromtxt('/kaggle/input/iris/Iris.csv', delimiter=',', skip_header=1, usecols = [1, 2, 3, 4])

iris[:,0][np.random.randint(0 , len(iris), 50)] = np.nan # set some random values in the first column

iris

nan_index_1 = np.where(np.isnan(iris)) # to check for nan, the official documentation always recommends using np.isnan

nan_index_1

iris[nan_index_1]

print("Number of missing values: \n", np.isnan(iris[:, 0]).sum())
# Q. Filter the rows of iris_2d that has petallength (3rd column) > 1.5 and sepallength (1st column) < 5.0



# Input

# Use the iris dataset provided

print_files()



# Use this if you are working on your local machine

# url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

# iris_2d = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0,1,2,3])



# Solution

print("Solution")

iris = np.genfromtxt('/kaggle/input/iris/Iris.csv', delimiter=',', skip_header=1, usecols = [1, 2, 3, 4])

reduce_array = iris[(iris[:,3] > 1.5) & (iris[:,1] < 5)]

iris.shape

reduce_array.shape

reduce_array



# Another solution

print("Using criteria saved as objects")

cond1 = iris[:,3] > 1.5

cond2 = iris[:,1] < 5

reduce_array2 = iris[cond1 & cond2]

reduce_array2.shape

reduce_array2



# Another solution using reduce

print("Using reduce")

from functools import reduce

criteria = reduce(lambda x, y: x & y, (cond1, cond2))

iris[criteria].shape

iris[criteria]
# Q. Select the rows of iris_2d that does not have any nan value.



# Input

# Use the titanic dataset provided

print_files()



# set original print statements

np.set_printoptions(edgeitems=3,infstr='inf', linewidth=75, nanstr='nan', precision=8, suppress=False, threshold=1000, formatter=None)



# Solution

print("Solution")



# Importing the titanic df

def import_titanic():

    with open("/kaggle/input/titanic/train.csv", "r") as f:

        data = f.read()

        l = []

        for row in data.split("\n")[1:-1]:

            r_ = row.split(",")

            l_ = []

            for c in r_:

                if c == "": 

                    l_.append(np.nan)

                else:

                    try:

                        l_.append(float(c))

                    except:

                        l_.append(c)

            l.append(l_)

    return l



l = import_titanic()

# only numeric columns

a = np.array(l, dtype = object)[:,[1, 2, 6, 7, 8, 10]]

# convert to float

arr = np.array(a, dtype = float)

# select rows with nan values

nan_r = np.array([~np.any(np.isnan(row)) for row in arr])

# filter the array

arr_no_nan = arr[nan_r]

arr_no_nan

# check: the sum of nans must be zero

np.isnan(arr_no_nan).sum()



# Solution 2

print("Solution from the website")

l = import_titanic()

a = np.array(l, dtype = object)[:,[1, 2, 6, 7, 8, 10]]

arr = np.array(a, dtype = float)

arr[np.sum(np.isnan(arr), axis = 1) == 0] # much more elegant solution

np.isnan(arr[np.sum(np.isnan(arr), axis = 1) == 0]).sum()
# Q. Find the correlation between SepalLength(1st column) and PetalLength(3rd column) in iris_2d



# Input

# Use the iris dataset provided

print_files()



# Use this if you are working on your local machine

# url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

# iris_2d = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0,1,2,3])



# Solution

print("Solution")

iris = np.genfromtxt('/kaggle/input/iris/Iris.csv', delimiter=',', skip_header=1, usecols = [1, 2, 3, 4])

np.correlate(iris[:,1], iris[:,3]) # This function computes the correlation as generally defined in signal processing texts: c_{av}[k] = sum_n a[n+k] * conj(v[n])

np.corrcoef(iris[:,1], iris[:,3]) # Pearson correlation



# Solution from the website

print("Solution from the website using scipy")

from scipy.stats.stats import pearsonr  

corr, p_value = pearsonr(iris[:, 1], iris[:, 3])

print(corr)

print(p_value)
# Q. Find out if iris_2d has any missing values.



# Input

# Use the iris dataset provided

print_files()



# Use this if you are working on your local machine

# url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

# iris_2d = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0,1,2,3])



# Solution

print("Solution")

iris = np.genfromtxt("/kaggle/input/iris/Iris.csv", delimiter=",",  dtype='float', usecols=[1,2,3,4], skip_header=1)

iris[np.random.randint(150, size=20), np.random.randint(4, size=20)] = np.nan # inser some null values



print("It's {} that we have nan values. The total amout of nan values is {}".format(np.any(np.isnan(iris)), np.isnan(iris).sum())) # first returns True second the total of nan values

# Q. Replace all ccurrences of nan with 0 in numpy array



# Input

# Use the iris dataset provided

print_files()



# Use this if you are working on your local machine

# url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

# iris_2d = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0,1,2,3])

# iris_2d[np.random.randint(150, size=20), np.random.randint(4, size=20)] = np.nan



# Solution

print("Solution")

iris = np.genfromtxt("/kaggle/input/iris/Iris.csv", delimiter=",",  dtype='float', usecols=[1,2,3,4], skip_header=1)

iris[np.random.randint(150, size=20), np.random.randint(4, size=20)] = np.nan # inser some null values



print("Before applying nan_to_num.")

np.isnan(iris).sum()

a = np.nan_to_num(iris, 0)

print("After applying nan_to_num.")

np.isnan(a).sum()



# Solution from the website

print("Solution from the website")

iris = np.genfromtxt("/kaggle/input/iris/Iris.csv", delimiter=",",  dtype='float', usecols=[1,2,3,4], skip_header=1)

iris[np.random.randint(150, size=20), np.random.randint(4, size=20)] = np.nan # inser some null values

np.isnan(iris).sum()

iris[np.isnan(iris)] = 0

np.isnan(a).sum()
# Q. Find the unique values and the count of unique values in iris's species



# Input

# Use the iris dataset provided

print_files()



# Use this if you are working on your local machine

# url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

# iris_2d = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0,1,2,3])

# names = ('sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'species')



# Solution

print("Solution using list comprehension")

iris = np.genfromtxt("/kaggle/input/iris/Iris.csv", delimiter = ",", usecols=[5], dtype=object, skip_header=1)

l = [(v, np.count_nonzero(iris[iris == v])) for v in np.unique(iris)]

l



# Solution from the website

print("Solution from the website")

iris = np.genfromtxt("/kaggle/input/iris/Iris.csv", delimiter = ",", usecols=[5], dtype=object, skip_header=1)

np.unique([v for v in iris], return_counts=True) # much more elegant
# Q. Bin the petal length (3rd) column of iris_2d to form a text array, such that if petal length is:

# Less than 3 --> 'small'

# 3-5 --> 'medium'

# '>=5 --> 'large'



# Input

# Use the iris dataset provided

print_files()



# Use this if you are working on your local machine

# url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

# iris_2d = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0,1,2,3])

# names = ('sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'species')



# Solution

print("Solution")

iris = np.genfromtxt("/kaggle/input/iris/Iris.csv", delimiter = ",", usecols=[3], dtype=float, skip_header=1)

bin_ = np.digitize(iris.astype('float'), [0, 3, 5, 10])

bin_

label_map = {1: 'small', 2: 'medium', 3: 'large', 4: np.nan}

cat_ = [label_map[x] for x in bin_]

cat_[:5]
# Q. Create a new column for volume in iris_2d, where volume is (pi x petallength x sepal_length^2)/3



# Input

# Use the iris dataset provided

print_files()



# Use this if you are working on your local machine

# url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

# iris_2d = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0,1,2,3])

# names = ('sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'species')



# Solution

print("Solution: use numpy.c_[] for columns")

iris = np.genfromtxt("/kaggle/input/iris/Iris.csv", delimiter = ",", usecols=[2, 3], dtype=float, skip_header=1)

iris.shape

iris = np.c_[iris, (np.array(iris[:,0] * 3.14 * (iris[:,1])**2))/3]

iris.shape

iris[:5]



# Solution from the website

print("Solution from website")

iris = np.genfromtxt("/kaggle/input/iris/Iris.csv", delimiter = ",", usecols=[2, 3], dtype=float, skip_header=1)

s = iris[:, 1]

p = iris[:, 0]

volume = (np.pi * p * (s**2))/3

# Introduce new dimension to match iris_2d's

volume = volume[:, np.newaxis]

# Add the new column

out = np.hstack([iris, volume])

# View

out[:4]
# Q. Randomly sample iris's species such that setose is twice the number of versicolor and virginica



# Input

# Use the iris dataset provided

print_files()



# Use this if you are working on your local machine

# url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

# iris = np.genfromtxt(url, delimiter=',', dtype='object')



# Solution from the website

print("Solution from the website")

iris = np.genfromtxt("/kaggle/input/iris/Iris.csv", delimiter = ",", dtype=object, skip_header=1)

# Get the species column

species = iris[:, 5]



# Approach 1: Generate Probablistically

print("Solution 1: generate probablistically")

np.random.seed(100)

a = np.array(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])

species_out = np.random.choice(a, 150, p=[0.5, 0.25, 0.25])

species_out



# Approach 2: Probablistic Sampling (preferred)

print("Solution 2: probablistic sampling")

np.random.seed(100)

probs = np.r_[np.linspace(0, 0.500, num=50), np.linspace(0.501, .750, num=50), np.linspace(.751, 1.0, num=50)]

index = np.searchsorted(probs, np.random.random(150))

species_out = species[index]

print(np.unique(species_out, return_counts=True))
# Q. What is the value of second longest petallength of species setosa



# Input

# Use the iris dataset provided

print_files()



# Use this if you are working on your local machine

# url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

# iris_2d = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0,1,2,3])

# names = ('sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'species')



# Solution

print("Solution")

headers = np.genfromtxt("/kaggle/input/iris/Iris.csv", delimiter = ",", dtype=object, skip_header=0)

headers[0]

iris = np.genfromtxt("/kaggle/input/iris/Iris.csv", delimiter = ",", dtype=object, skip_header=1)

# iris

sorted_iris = iris[iris[:,5] == b'Iris-setosa'][:,3]

sorted_iris = sorted_iris.astype(float)

sorted_iris = np.unique(sorted_iris)

sorted_iris.sort()

sorted_iris[::-1][1]



# Solution from the website

print("Solution from the website")

iris = np.genfromtxt("/kaggle/input/iris/Iris.csv", delimiter = ",", dtype=object, skip_header=1)

petal_len_setosa = iris[iris[:, 5] == b'Iris-setosa', [3]].astype('float')

np.unique(np.sort(petal_len_setosa))[-2]
# Q. Sort the iris dataset based on sepallength column.



# Input

# Use the iris dataset provided

print_files()



# Use this if you are working on your local machine

# url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

# iris = np.genfromtxt(url, delimiter=',', dtype='object')

# names = ('sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'species')



# Solution

print("Solution")

headers = np.genfromtxt("/kaggle/input/iris/Iris.csv", delimiter = ",", dtype=object, skip_header=0)

print(list(headers[0]))

iris = np.genfromtxt("/kaggle/input/iris/Iris.csv", delimiter = ",", usecols=[0,1,2,3,4], skip_header=1)

iris[iris[:,1].argsort()][:5]



# Solution from the website

print("Solution from the website")

print(iris[iris[:,1].argsort()][:5]) # same solution
# Q. Find the most frequent value of petal length (3rd column) in iris dataset.



# Input

# Use the iris dataset provided

print_files()



# Use this if you are working on your local machine

# url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

# iris = np.genfromtxt(url, delimiter=',', dtype='object')

# names = ('sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'species')



# Solution

np.set_printoptions(edgeitems=3,infstr='inf', linewidth=75, nanstr='nan', precision=3, suppress=False, threshold=1000, formatter=None)

print("Solution")

headers = np.genfromtxt("/kaggle/input/iris/Iris.csv", delimiter = ",", dtype=object, skip_header=0)

print(list(headers[0]))

iris = np.genfromtxt("/kaggle/input/iris/Iris.csv", delimiter = ",", usecols=[4], skip_header=1)

counts = np.unique([v for v in iris], return_counts=True) # we have a tuple of arrays

sort_list = sorted(list(zip(counts[0], counts[1])), key = lambda x: x[1]) # extract the values and counts, zip them and sort by the counts

sort_list[::-1][0] #reverse the list and get the first (most frequent) the most frequent value of petal lenght is 0.2, it has ocurred 28 times



# Solution from the website

print("Solution from the website")

vals, counts = np.unique(iris, return_counts=True)

print(vals[np.argmax(counts)]) # much more elegant solution
# Q. Find the position of the first occurrence of a value greater than 1.0 in petalwidth 4th column of iris dataset.



# Input

# Use the iris dataset provided

print("Input")

print_files()



# Use this if you are working on your local machine

# url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

# iris = np.genfromtxt(url, delimiter=',', dtype='object')



# Solution

print("Solution")

iris = np.genfromtxt("/kaggle/input/iris/Iris.csv", delimiter = ",", usecols=[4], skip_header=1)

i = np.where(iris > 1) # returns the index where this is tru

i

iris[i[0][0]] # first ocurrence is on index 50

iris[:i[0][0] + 1]



# Solution from the website

print("Solution from website")

np.argwhere(iris.astype(float) > 1.0)[0] # same result but much faster and elegant
# Q. From the array a, replace all values greater than 30 to 30 and less than 10 to 10.

# Input

print("Input")

np.random.seed(100)

a = np.random.uniform(1,50, 20)

a



# Solution

print("Solution")

c1 = np.where(a > 30)

c2 = np.where(a < 10)

a[c1] = 30

a[c2] = 10

a



# Solution from the website

print("Solution from the website")



# Solution 1: Using np.clip

np.random.seed(100)

a = np.random.uniform(1,50, 20)

np.clip(a, a_min=10, a_max=30) # probabily the most elegant solution



# Solution 3: Using np.where

np.random.seed(100)

a = np.random.uniform(1,50, 20)

print(np.where(a < 10, 10, np.where(a > 30, 30, a)))
# Q. Get the positions of top 5 maximum values in a given array a.



# Input

print("Input")

np.random.seed(100)

a = np.random.uniform(1,50, 20)

a



# Solution

print("Solution")

a.argsort() # sort the numpy array with argsort(), returns the index starting from min to max value. Index 15 has the max value

a.argsort()[-5:] # select top 5 index

a.argsort()[-5:][::-1] # a.argsort()[::-1][:5] are equivalent, reverse the top 5 indee

a[a.argsort()[-5:][::-1]] # get the values



# Solution from the website

print("Solution from the webpage")

# Solution:

print(a.argsort())

#> [18 7 3 10 15]



# Solution 2:

np.argpartition(-a, 5)[:5]

#> [15 10  3  7 18]



# Below methods will get you the values.

# Method 1:

a[a.argsort()][-5:]



# Method 2:

np.sort(a)[-5:]



# Method 3:

np.partition(a, kth=-5)[-5:]



# Method 4:

a[np.argpartition(-a, 5)][:5]
# Q. Compute the counts of unique values row-wise.



# Input

print("Input")

np.random.seed(100)

arr = np.random.randint(1,11,size=(6, 10))

arr



# Desired Output

# > [[1, 0, 2, 1, 1, 1, 0, 2, 2, 0],

# >  [2, 1, 3, 0, 1, 0, 1, 0, 1, 1],

# >  [0, 3, 0, 2, 3, 1, 0, 1, 0, 0],

# >  [1, 0, 2, 1, 0, 1, 0, 2, 1, 2],

# >  [2, 2, 2, 0, 0, 1, 1, 1, 1, 0],

# >  [1, 1, 1, 1, 1, 2, 0, 0, 2, 1]]



# Output contains 10 columns representing numbers from 1 to 10. The values are the counts of the numbers in the respective rows.

# For example, Cell(0,2) has the value 2, which means, the number 3 occurs exactly 2 times in the 1st row.



# Solution

print("Solution")

from collections import Counter

rows = arr.shape[0]

lc = []

for row in range(rows): # iterate over all rows in the numpy array

    counter = Counter() # on every row, create a new Counter

    counter.update(arr[row]) # feed the Counter with the row

    lc.append(dict(counter)) # append a dict in a list: each row will have it's unique dict/counter



np.array([np.vectorize(lc[row].get)(arr[row]) for row in range(rows)]) # trasnfrom the arr into a vector and map to the values in the dictionary (lc[row].get gets every dictionary) for each row in rows



# more interesting aproaches here:

# https://stackoverflow.com/questions/16992713/translate-every-element-in-numpy-array-according-to-key



# Solution from the website

print("Solution from the website (incorrect)")



def counts_of_all_values_rowwise(arr2d):

    # Unique values and its counts row wise

    num_counts_array = [np.unique(row, return_counts=True) for row in arr2d]

    # Counts of all values row wise

    return([[int(b[a==i]) if i in a else 0 for i in np.unique(arr2d)] for a, b in num_counts_array])



# it has a bug

print("The solution from the website has a bug.")

counts_of_all_values_rowwise(arr)



def counts_of_all_values_rowwise_corrected(arr2d):

    num_counts_array = [np.unique(row, return_counts=True) for row in arr2d] # same as in the previous solution

    # we have a numpy array of tuples that contain for each row the elements and their counts

    ll = [] # create an empty list of lists, we will later convert in into numpy array

    for i in range(arr2d.shape[0]): # for each row in arr rows

        l = [] # create a new list where we will be adding the mappings

        rmapper = num_counts_array[i] # rmapper = row mapper. In num_counts_array for have the same amount of tuples (elemnt - count) as rows in the arr2d

        for v in arr2d[i]: # for each value in a row

            l.append(rmapper[1][np.where(rmapper[0] == v)][0]) # append to the list the count (rmapper[1]), we are using np.where to find the index of the element

        ll.append(l) # append the list to the list of lists

    return np.array(ll) # convert the lst of list into numpy 2d array



print("The solution from the website corrected.")

counts_of_all_values_rowwise_corrected(arr)
# Q. Convert array_of_arrays into a flat linear 1d array.



# Input

print("Input")

arr = np.arange(9).reshape(3,3)

arr



# Solution

print("Solution")

arr.flatten()



# Solution from the website

arr1 = np.arange(3)

arr2 = np.arange(3,7)

arr3 = np.arange(7,10)



array_of_arrays = np.array([arr1, arr2, arr3])

print('array_of_arrays: ', array_of_arrays)



# Solution 1

arr_2d = np.array([a for arr in array_of_arrays for a in arr]) # interesting loop comprehension

arr_2d



print("------------------")

arr_2d = []

for arr in array_of_arrays:

    for a in arr:

        arr_2d.append(a)

np.array(arr_2d)





# Solution 2:

arr_2d = np.concatenate(array_of_arrays)

print(arr_2d)
# Q. Compute the one-hot encodings (dummy binary variables for each unique value in the array)



# Input

print("Input")

np.random.seed(101) 

arr = np.random.randint(1,4, size=6)

arr



# Desired Output

# > array([[ 0.,  1.,  0.],

# >        [ 0.,  0.,  1.],

# >        [ 0.,  1.,  0.],

# >        [ 0.,  1.,  0.],

# >        [ 0.,  1.,  0.],

# >        [ 1.,  0.,  0.]])



# Solution

print("Solution using pandas")

import pandas as pd

df = pd.DataFrame(arr)

dummies = pd.get_dummies(df[0])

np.array(dummies)



# Solution using pure python

print("Solution using pure python")

ll = []

for i in list(set(arr)):

    l = []

    for j in arr:

        l.append(1) if i == j else l.append(0)

    ll.append(l)

np.array(ll).T



# Solution using pure python with list comprehension

print("Solution using pure python with list comprehension (list of lists)")

np.array([[1 if i == j else 0 for i in list(set(arr))] for j in arr])



# Solution using numpy

# Solution from the website

print("Solution from the website")

print("Solution 1 using numpy")



uniques = np.unique(arr)

out = np.zeros((arr.shape[0], uniques.shape[0]))

for i, k in enumerate(arr):

    print(i, k)

    out[i, k-1] = 1 # very cool solution

out



print("Solution 2 using numpy")

print("arr[:, None] evaluates all the numpy array to the unique elements and returns True or False")

(arr[:,None] == np.unique(arr))

print("we add .view(np.int8) to convert Boolean to 1 or zero")

(arr[:,None] == np.unique(arr)).view(np.int8)

# Q. Create row numbers grouped by a categorical variable. Use the following sample from iris species as input.



# Input

# Use the iris dataset provided

print("Input")

print_files()



# Use this if you are working on your local machine

# species = np.genfromtxt(url, delimiter=',', dtype='str', usecols=4)

# species_small = np.sort(np.random.choice(species, size=20))

# species_small

# > array(['Iris-setosa', 'Iris-setosa', 'Iris-setosa', 'Iris-setosa',

# >        'Iris-setosa', 'Iris-setosa', 'Iris-versicolor', 'Iris-versicolor',

# >        'Iris-versicolor', 'Iris-versicolor', 'Iris-versicolor',

# >        'Iris-versicolor', 'Iris-virginica', 'Iris-virginica',

# >        'Iris-virginica', 'Iris-virginica', 'Iris-virginica',

# >        'Iris-virginica', 'Iris-virginica', 'Iris-virginica'],

# >       dtype='<U15')





# Desired Output

# > [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 7]



# Solution

print("Solution")

species = np.genfromtxt("/kaggle/input/iris/Iris.csv", delimiter=',', dtype='str', usecols=5, skip_header=1)

species_small = np.sort(np.random.choice(species, size=20))

species_small

ll = [[i for i in range(len(species_small[species_small == j]))] for j in np.unique(species_small)] # create a list of lits

[i for l in ll for i in l] # flatten the list: read from the first for: for list in lists for i in list append i. THE FIRST i

# https://stackoverflow.com/questions/952914/how-to-make-a-flat-list-out-of-list-of-lists



# Solution from the website

print("Solution from the website")

[i for val in np.unique(species_small) for i, grp in enumerate(species_small[species_small==val])]
# Q. Create group ids based on a given categorical variable. Use the following sample from iris species as input



# Input

# Use the iris dataset provided

print("Input")

print_files()



# Use this if you are working on your local machine

# species = np.genfromtxt(url, delimiter=',', dtype='str', usecols=4)

# species_small = np.sort(np.random.choice(species, size=20))

# species_small

# > array(['Iris-setosa', 'Iris-setosa', 'Iris-setosa', 'Iris-setosa',

# >        'Iris-setosa', 'Iris-setosa', 'Iris-versicolor', 'Iris-versicolor',

# >        'Iris-versicolor', 'Iris-versicolor', 'Iris-versicolor',

# >        'Iris-versicolor', 'Iris-virginica', 'Iris-virginica',

# >        'Iris-virginica', 'Iris-virginica', 'Iris-virginica',

# >        'Iris-virginica', 'Iris-virginica', 'Iris-virginica'],

# >       dtype='<U15')



# Desired Output

# > [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]



# Solution

print("Solution")

species = np.genfromtxt("/kaggle/input/iris/Iris.csv", delimiter=',', dtype='str', usecols=5, skip_header=1)

species_small = np.sort(np.random.choice(species, size=20))

species_small



d = dict((k, i) for i, k in enumerate(np.unique(species_small))) # create a mapping for every specie and store it in a dictionary

np.array([np.vectorize(d.get)(sp) for sp in species_small]) # use vectorize and dictionary .get method to map all the species



# Solution from the website

print("Solution from the website")

print("Solution usig numpy")

output = [np.argwhere(np.unique(species_small) == s).tolist()[0][0] for val in np.unique(species_small) for s in species_small[species_small==val]]

output



# Solution: For Loop version

print("Solution using for loops")

output = []

uniqs = np.unique(species_small)



for val in uniqs:  # uniq values in group

    for s in species_small[species_small==val]:  # each element in group

        groupid = np.argwhere(uniqs == s).tolist()[0][0]  # groupid

        output.append(groupid)

output
# Q. Create the ranks for the given numeric array a.



# Input

print("Input")

np.random.seed(10)

a = np.random.randint(20, size=10)

a



# Desired Output

# [4 2 6 0 8 7 9 3 5 1]



# Solution

print("Solution")

a.argsort().argsort() # use argosrt twice: first to find the order of the array and then the rank

print("Order of the array: in the index 3 of the array (a) we have the value 0, which is the smallest value - ")

print("- at the index 9, we have the value 0 which is the second largest value etc etc etc")

a.argsort()

print("Second argsort: the value 9 in the original array is the 4th smallest value, the value 4 in the original array is the second smallest value etc etc etc")

a.argsort().argsort



# reference: https://stackoverflow.com/questions/5284646/rank-items-in-an-array-using-python-numpy-without-sorting-array-twice/
# Q. Create a rank array of the same shape as a given numeric array a.



# Input

print("Input")

np.random.seed(10)

a = np.random.randint(20, size=[2,5])

a



# Desired Output

# > [[4 2 6 0 8]

# >  [7 9 3 5 1]]



# Solution

print("Solution")

a.flatten().argsort().argsort().reshape(2, -1) # flatten first the array, then use the same tecnique as before and then reshape the array



# Solution from the website

print("Solution from the website")

print(a.ravel().argsort().argsort().reshape(a.shape))



'''

Difference between flatten and ravel:



- flatten is a method of an ndarray object and hence can only be called for true numpy arrays.



- ravel is a library-level function and hence can be called on any object that can successfully be parsed.

'''
# Q. Compute the maximum for each row in the given array.



# Input

print("Input")

np.random.seed(100)

a = np.random.randint(1,10, [5,3])

a



# Solution

print("Solution")

np.array([max(row) for row in a])



# Solution from the website

print("Solution from the website")



# Solution 1

np.amax(a, axis=1)



# Solution 2

np.apply_along_axis(np.max, arr=a, axis=1)
# Q. Compute the min-by-max for each row for given 2d numpy array.



# Input

print("Input")

np.random.seed(100)

a = np.random.randint(1,10, [5,3])

a



# Solution

print("Solution")

np.array([min(row)/max(row) for row in a])



# Solution from the website

print("Solution from the website")

np.apply_along_axis(lambda x: np.min(x)/np.max(x), arr=a, axis=1) # maybe a little more elegat solution
# Q. Find the duplicate entries (2nd occurrence onwards) in the given numpy array and mark them as True. First time occurrences should be False.



# Input

print("Input")

np.random.seed(100)

a = np.random.randint(0, 5, 10)

a



# Desired Output

# > [False  True False  True False False  True  True  True  True]



# Solution

print("Solution")

print("Ooops, I have not understood the problem. This solution marks False the elements that are duplicated and True the unique ones.")

counts = np.unique(a, return_counts=True)

np.array([True if counts[1][np.where(counts[0] == x)] > 1 else False for x in a])



# Solution from the website

print("Solution from the website")

out = np.full(a.shape[0], True) # Create an all True array

unique_positions = np.unique(a, return_index=True)[1] # Find the index positions of unique elements

out[unique_positions] = False # Mark those positions as False

out
# Q. Find the mean of a numeric column grouped by a categorical column in a 2D numpy array



# Input

# Use the iris dataset provided

print("Input")

print_files()



# Use this if you are working on your local machine

# url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

# iris = np.genfromtxt(url, delimiter=',', dtype='object')

# names = ('sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'species')



# Desired Output

# > [[b'Iris-setosa', 3.418],

# >  [b'Iris-versicolor', 2.770],

# >  [b'Iris-virginica', 2.974]]



# Solution

print("Solution")

iris = np.genfromtxt("/kaggle/input/iris/Iris.csv", delimiter=',', dtype='object', skip_header=1)

sepallength = iris[:,2].astype(float)

names = iris[:,5]

[[name, np.mean(sepallength[np.where(names == name)])] for name in np.unique(names)]



# Solution from the website

print("Solution from the website")

numeric_column = iris[:, 2].astype('float')  # sepalwidth

grouping_column = iris[:, 5]  # species



# List comprehension version

[[group_val, numeric_column[grouping_column==group_val].mean()] for group_val in np.unique(grouping_column)]



# For Loop version

output = []

for group_val in np.unique(grouping_column):

    output.append([group_val, numeric_column[grouping_column==group_val].mean()])



output

# Q. Import the image from the following URL and convert it to a numpy array.



# Input

print("Input")

print_files()



# Use this if you are working on your local machine

# URL = 'https://upload.wikimedia.org/wikipedia/commons/8/8b/Denali_Mt_McKinley.jpg'



# Solution

print("Solution")

from PIL import Image

# pic = Image.open("/kaggle/input/exercise-60-denali-mt-mckinleyjpg/Denali Mt McKinley.jpg").convert("L") # works also without convert("L")

pic = Image.open("/kaggle/input/exercise-60-denali-mt-mckinleyjpg/Denali Mt McKinley.jpg")

imgarr = np.array(pic) 

imgarr



# Solution from the website

print("Solution from the website")

from io import BytesIO

from PIL import Image

import PIL



# Read it as Image

I = Image.open("/kaggle/input/exercise-60-denali-mt-mckinleyjpg/Denali Mt McKinley.jpg")



# Optionally resize

I = I.resize([150,150])



# Convert to numpy array

arr = np.asarray(I)

arr



# Optionaly Convert it back to an image and show

im = PIL.Image.fromarray(np.uint8(arr))

Image.Image.show(im)
# Q. Drop all nan values from a 1D numpy array



# Input

print("Input")

a = np.array([1,2,3,np.nan,5,6,7,np.nan])

a



# Desired Output

# array([ 1.,  2.,  3.,  5.,  6.,  7.])



# Solution

print("Solution")

a[[not np.isnan(x) for x in a]] # create a list with boolean if its np.isnan (not to reverse to True when it's not a nan) and index the original array



# Solution from the website

print("Solution from the website")

a[~np.isnan(a)]
# Q. Compute the euclidean distance between two arrays a and b.



# Input

print("Input")

a = np.array([1,2,3,4,5])

b = np.array([4,5,6,7,8])

a

b



# Solution

print("Solution")

dist = np.linalg.norm(a-b)

dist



# The website uses the same solution
# Q. Find all the peaks in a 1D numpy array a. Peaks are points surrounded by smaller values on both sides.



# Input

print("Input")

a = np.array([1, 3, 7, 1, 2, 6, 0, 1])

a



# Desired Output

# > array([2, 5])

# where, 2 and 5 are the positions of peak values 7 and 6.



# Solution

print("Solution using scipy")

from scipy.signal import find_peaks



ipeaks, _ = find_peaks(a)

ipeaks

a[ipeaks]



# Solution from the website

print("Solution from the website using pure numpy")

doublediff = np.diff(np.sign(np.diff(a)))

peak_locations = np.where(doublediff == -2)[0] + 1

peak_locations
# Q. Subtract the 1d array b_1d from the 2d array a_2d, such that each item of b_1d subtracts from respective row of a_2d.



# Input

print("Input")

a_2d = np.array([[3,3,3],[4,4,4],[5,5,5]])

b_1d = np.array([1,2,3])

a_2d

b_1d



# Desired Output

# > [[2 2 2]

# >  [2 2 2]

# >  [2 2 2]]



# Solution

print("Solution")

a_2d - b_1d[:, None]
# Q. Find the index of 5th repetition of number 1 in x.



# Input

print("Input")

x = np.array([1, 2, 1, 1, 3, 4, 3, 1, 1, 2, 1, 1, 2])

x



# Solution

print("Solution")

np.where(x == 1)[0][4] # we do [0] since np.where returns a tupple, and then find the index of the 5 repetition



# Solution from the website

print("Solution from the website")

n = 5

[i for i, v in enumerate(x) if v == 1][n-1]



print("Solution using numpy")

np.where(x == 1)[0][n-1] # notice that n = 5, and 5 - 1 = 4, the index we did
# Q. Convert numpy's datetime64 object to datetime's datetime object



# Input

print("Input")

dt64 = np.datetime64('2018-02-25 22:10:10')

dt64



# Solution from the website

print("Solution from the website")

from datetime import datetime

dt64.tolist()



# or



dt64.astype(datetime)
# Q. Compute the moving average of window size 3, for the given 1D array.



# Input

print("Input")

np.random.seed(100)

a = np.random.randint(10, size=10)

a



# Solution

print("Solution")

# using the solution from https://stackoverflow.com/questions/14313510/how-to-calculate-moving-average-using-numpy/54628145

def moving_average(a, n=3) :

    ret = np.cumsum(a, dtype=float)

    ret[n:] = ret[n:] - ret[:-n]

    return ret[n - 1:] / n



moving_average(a)



# Solution from the website

print("Solution from the website")

np.convolve(a, np.ones(3)/3, mode='valid')
# Q. Create a numpy array of length 10, starting from 5 and has a step of 3 between consecutive numbers



# Solution

print("Solution")



np.arange(5, (5 + (10*3)), 3) # first argument is the starting point, second is the end, and the third the step.
# Q. Given an array of a non-continuous sequence of dates. Make it a continuous sequence of dates, by filling in the missing dates.



# Input

print("Input")

dates = np.arange(np.datetime64('2018-02-01'), np.datetime64('2018-02-25'), 2)

dates



# Solution from the website

print("Solution from the website")

# Solution ---------------

filled_in = np.array([np.arange(date, (date+d)) for date, d in zip(dates, np.diff(dates))]).reshape(-1)



# add the last day

output = np.hstack([filled_in, dates[-1]])

output



# For loop version -------

out = []

for date, d in zip(dates, np.diff(dates)):

    out.append(np.arange(date, (date+d)))



filled_in = np.array(out).reshape(-1)



# add the last day

output = np.hstack([filled_in, dates[-1]])

output
# Q. From the given 1d array arr, generate a 2d matrix using strides, with a window length of 4 and strides of 2, like [[0,1,2,3], [2,3,4,5], [4,5,6,7]..]



# Input

print("Input")

arr = np.arange(15) 

arr



# Desired Output

# > [[ 0  1  2  3]

# >  [ 2  3  4  5]

# >  [ 4  5  6  7]

# >  [ 6  7  8  9]

# >  [ 8  9 10 11]

# >  [10 11 12 13]]



# Solution

print("Solution")

index_ = np.arange(0, 15, 2)

arr_ = [[arr[index_[i]:index_[i+2]]] for i in range(6)]

arr_



# Solution from the website

print("Solution from the website")

def gen_strides(a, stride_len=5, window_len=5):

    

    n_strides = ((a.size-window_len)//stride_len) + 1

    

    return np.array([a[s:(s+window_len)] for s in np.arange(0, n_strides*stride_len, stride_len)])



print(gen_strides(np.arange(15), stride_len=2, window_len=4))