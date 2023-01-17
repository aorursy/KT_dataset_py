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
import warnings

warnings.filterwarnings('ignore')
# This will help us in truncating the longer array

np.set_printoptions(threshold = 100)
# for i in range(114, 128):

#     print(f'* [{i}. ](#{i})  ')
# 0. Numpy version



np.__version__
# 1. Create an empty array



empty_array = np.empty([2, 2], int)



print(empty_array)



print(empty_array.shape)



print(empty_array.size)
# 2. Check whether the array is empty



a = np.array([])

b = np.array([1, 2])



def check_empty(a):

    if a.size == 0:

        print(a, ' : Empty')

    else:

        print(b, ' : Non Empty')

        

check_empty(a)

check_empty(b)
# 3. Check elements count



c = np.array([])

d = np.array([1, 2])



def get_elements(c_array):

    return c_array.ndim and c_array.size

        

print(c, ', elements_count : ', get_elements(c))

print(d, ', elements_count : ', get_elements(d))
# 4. Arrange numpy between numbers



a = np.arange(4, 12)



print(a)

print(a, '.shape : ', a.shape)
# 5. Arrange numpy between numbers with intervals



c = np.arange(12, 30, 3)



print(c)

print(c, '.shape : ', c.shape)
# 6. Array reshape



c = np.arange(12, 30, 3)

d = c.reshape(2, 3)



print(c)

print(c, '.shape : ', c.shape)

print('\nAfter reshaping : ')

print(d)

print(d, '.shape : ', d.shape)
# 7. Create a numpy with random integers



a = np.random.randint(10, size = 5)



print(a)
# 8. Create a numpy with random integers and size



a = np.random.randint(10, size = (2, 3))



print(a)

print(a, '.shape : ', a.shape)

print('datatype : ', a.dtype)
# 9. Array of strings



a = np.array(('Toronto','Montreal','New York'))

print(a)

print(a.dtype)
# 10. Numpy array with strings and explicit dtype



x = np.array(['Toronto', 'Montreal'], dtype=str)



print(x)

print(x.dtype)
# 11. Numpy array with strings and explicit dtype



x = np.array(['To'], dtype=str)

y = x.view('S1').reshape(x.size, -1)



print(y)
# 12. Print without truncation



x = np.arange(10000)



print('Before setting print options : ')

print(x) # this will print with the truncation



# set

import sys

# np.set_printoptions(threshold = sys.maxsize)



# print('\nAfter setting print options : ')

# print(a) # this will print everything without truncation
# 13. Save Numpy to CSV



a = np.asarray([ [1,2,3], [4,5,6]])

print(a)

# numpy.savetxt("abc.csv", a, delimiter=",") # this will save the numpy to csv file
# 14. Dataframe to Numpy



df = pd.DataFrame({'a1': [1, 2, 3], 'a2': [4, 5, 6]}, index = ['X', 'Y', 'Z'])



print('Dataframe:')

print(df)



x = df.to_numpy()



print('\nDataframe to Numpy:')

print(x)



y = df.index.to_numpy()

print('\nDataframe Indices to Numpy:')

print(y)



z = df['a1'].to_numpy()

print('\nDataframe Series to Numpy:')

print(z)
# 15. Get the nth column of an array



x = np.array([[1, 2], [3, 4], [5, 6]])



print('Numpy array:')

print(x)



y = x[:,0]

print('\nx[:,0]:')

print(y)



z = x[:, 1]

print('\nx[:, 1]:')

print(z)



a = x[1,:]

print('\nx[1,:]:')

print(a)
# 16. Reshape with -1 (lazy option)



a = np.matrix([[1, 2, 3, 4], [5, 6, 7, 8], [10, 11, 12, 21]])

print('Original Numpy Array:')

print(a)

print('original shape: ', a.shape)



c = np.reshape(a, -1)

print('\nnp.reshape(a, -1):')

print(c)

print('new shape: ', c.shape)



d = np.reshape(a, (1, -1))

print('\nnp.reshape(a, (1, -1)):')

print(d)

print('new shape: ', d.shape)



e = np.reshape(a, (2, -1))

print('\nnp.reshape(a, (2, -1)):')

print(e)

print('new shape: ', e.shape)



f = np.reshape(a, (3, -1))

print('\nnp.reshape(a, (3, -1)):')

print(f)

print('new shape: ', f.shape)



g = np.reshape(a, (4, -1))

print('\nnp.reshape(a, (4, -1)):')

print(g)

print('new shape: ', g.shape)
# 17. Numpy with precision



x = np.random.random(10)



print('Original Array:')

print(x)



print('\nAfter setting presicion:')

np.set_printoptions(precision = 2)

print(x)



# resetting precision to default (8)

np.set_printoptions(precision = 8)
# 18. Argsort on Numpy array



a = np.random.randint(0, 10, (3,3))

print('Before : ')

print(a)



print('\nAfter : ')

b = a[a[: ,2].argsort()]

print(b)
# 19. Numpy view



x = np.array([(1, 2)], dtype=[('a', np.int8), ('b', np.int8)])

print(x)

print(x.dtype)



y = x.view(dtype=np.int16, type=np.matrix)

print(y)

print(y.dtype)
# 20. Using Flipud



import numpy as np 

import matplotlib.pyplot as plt

import matplotlib.image as mp_img



image_a = mp_img.imread('/kaggle/input/numpy-cheatsheet/cn_tower.jpg')



image_b = np.flipud(image_a)

plt.imshow(image_b)



# plt.savefig("/kaggle/input/numpy-cheatsheet/cn_tower_ver.jpg", dpi=200)

plt.show()
# 21. Numpy inverse



b = np.array([[2,3],[4,5]])

print('Before Inverse : ')

print(b)



c = np.linalg.inv(b)

print('After Inverse : ')

print(c)
# 22. Numpy Inverse



x = np.matrix([[10, 20], [60, 70]])

print('Before Inverse : ')

print(x)



print('After Inverse : ')

print(x.I)
# 23. Numpy compare



a = np.arange(12).reshape((3, 4))

print(a)



a_bool = a < 6

print(a_bool)
# 24. Numpy compare with nonzero count



a = np.arange(12).reshape((3, 4))

print('Before : ')

print(a)



print(np.count_nonzero(a > 5))



print(a % 3 == 1)

print(np.count_nonzero(a % 3 == 1))
# 25. Flip a numpy array by using flipud

# flipup = flip ud = up / down



a = np.arange(4).reshape(2, 2)

print('Before : ')

print(a)



print('\nAfter : ')

b = np.flipud(a)

print(b)



# Note: b returns a view 

print('Shared memory? :', np.shares_memory(a, b))
# 26. Flip a numpy array by using flipud without sharing the memory



a = np.arange(4).reshape(2, 2)

print('Before : ')

print(a)



print('\nAfter : ')

b = np.flipud(a).copy()

print(b)



# Note: b returns a view 

print('Shared memory? :', np.shares_memory(a, b))
# 27. Flip a numpy array by using fliplr (horizontally)



a = np.arange(10).reshape(2, 5)

print('Before : ')

print(a)



print('\nAfter : ')

b = np.fliplr(a)

print(b)
# 28. Flip a numpy array by using flip (both horizontally and vertically)



a = np.arange(10).reshape(2, 5)

print('Before : ')

print(a)



print('\nAfter : ')

b = np.flip(a)

print(b)
# 29. Flipping the numpy array using slices



a = np.arange(10).reshape(2, 5)

print('Before : ')

print(a)



print('\nAfter : ')

b = a[::-1, ::-1]

print(b)
# 30. Convert numpy array to list



a = np.arange(10).reshape(2, 5)

print('Before : ')

print(a)

# print(d.dtype)

print(type(a))

print(type(a[0]))

print(type(a[0][0]))



b = a.tolist()

print('\nAfter : ')

print(b)

print(type(b))

print(type(b[0]))

print(type(b[0][0]))
# 31. Numpy Where



a = np.arange(8).reshape((2, 4))

print('Before : ')

print(a)



print('\nAfter : ')

b = np.where(a < 4, 0, 20)

print(b)



# Note: it matches, replace with 0, if not replace with 20
# 32. Numpy where with multiple conditions



a = np.arange(8).reshape((2, 4))

print('Before : ')

print(a)



print('\nAfter : ')

b = np.where((a > 3) & (a < 7), 0, 20)

print(b)
# 33. Numpy where with multiple conditions - apply only on matching conditions



a = np.arange(10).reshape((2, 5))

print('Before : ')

print(a)



print('\nAfter : ')

b = np.where((a > 3) & (a < 7), 0, a)

print(b)
# 34. Process with where



a = np.arange(10).reshape((2, 5))

print('Before : ')

print(a)



print('\nAfter : ')

b = np.where((a > 3) & (a < 7),  a * 3, 0)

print(b)
# 35. List to numpy array



a = [0, 1, 2]

print('Before : ')

print(a)



print('\nAfter : ')

b = np.array(a)

print(b)

print(b.dtype)
# 36. List to numpy array with explicit dtype



a = [0, 1, 2]

print('Before : ')

print(a)



print('\nAfter : ')

b = np.array(a, dtype = float)

print(b)

print(b.dtype)
# 37. 2D list to numpy array



a = [[0, 1, 2], [21, 22, 23]]

print('Before : ')

print(a)

print(type(a))



print('\nAfter : ')

b = np.array(a)

print(b)

print(type(b))

print(b.dtype)

print(b.shape)
# 38. Convert list to float numpy array



x = [1, 2]

print('Before : ')

print(x)

print(type(x))



print('\nAfter : ')

b = np.asfarray(x)

print(b)

print(type(b))

print(b.dtype)



print('\nAfter : ')

c = np.asarray(x, float)

print(c)

print(type(c))

print(c.dtype)
# 39. Convert list to numpy array with explicit datatype



x = [1, 2]

print('Before : ')

print(x)

print(type(x))



print('\nAfter : ')

c = np.asarray(x, float)

print(c)

print(type(c))

print(c.dtype)
# 40. Find common values between two numpy array



a = np.random.randint(0, 10, 10)

b = np.random.randint(0, 10, 10)

print(a)

print(b)

print('common values between a and b : ', np.intersect1d(a,b))
# 41. Get today in numpy and deltas



today = np.datetime64('today', 'D')

print('today          : ', today)



after2days = np.datetime64('today', 'D') + np.timedelta64(2, 'D')

print('after 2 days   : ', after2days)



before3days = np.datetime64('today', 'D') - np.timedelta64(3, 'D')

print('before 3 days  : ', before3days)



after1week = np.datetime64('today', 'D') + np.timedelta64(1, 'W')

print('after 1 week   : ', after1week)



after10weeks = np.datetime64('today', 'D') + np.timedelta64(10, 'W')

print('after 10 weeks : ', after10weeks)
# 42. Between two Dates



a = np.arange('2020-09-15', '2020-09-25', dtype='datetime64[D]')

print('Between two dates : 2020-09-15 and 2020-09-25')

print(a)



a = np.arange('2020-09', '2020-10', dtype='datetime64[D]')

print('\nBetween 2 months : 2020-09 and 2020-10')

print(a)
# 43. Random array and sorting



a = np.random.random(5)



print('Before : ')

print(a)



a.sort()

print('\nAfter : ')

print(a)
# 44. Random int array and sorting



a = np.random.randint(50, 100, 5)



print('Before : ')

print(a)



a.sort()

print('\nAfter : ')

print(a)
# 45. String to nump,y array



from io import StringIO



content = StringIO('''

1, 2, 3

6, ,  8

20, , 20

''')

a = np.genfromtxt(content, delimiter=",", dtype=np.int)

print(a)
# 46. Find the nearest element in the array



a = np.arange(10, 60, 7)

print(a)



b = 23

c = a.flat[np.abs(a - b).argmin()]

print(f'Elemenet near by {b} : {c}')
# 47. Swap rows



a = np.arange(9).reshape(3, 3)

print('Before : ')

print(a)



a[[0,1]] = a[[1, 0]]



print('\nAfter : ')

print(a)
# 48. Shuffle



a = np.arange(20)

print('Before : ')

print(a)



np.random.shuffle(a)



print('\nAfter : ')

print(a)
# 49. Get specific element



a = np.arange(27).reshape(3, 3, 3)

print(a)



print(a[0, 1, 1])
# 50. Repeat an array



a = np.array([[1, 2, 3]])

print('Before : ')

print(a)



b = np.repeat(a, 3, axis=0)

print('\nAfter : ')

print(b)
# 51. Min, Max, Sum



a = np.arange(6).reshape(2, 3)

a += 1

print(a)



a_mean = np.min(a)

print('Mean : ', a_mean)



a_max = np.max(a)

print('Max : ', a_max)



a_sum = np.sum(a)

print('Sum : ', a_sum)
# 52. Get min of axis = 1



x = np.arange(10).reshape((2, 5))



print('x:')

print(x)



print('\nx.min(axis = 1) : ')

print(x.min(axis = 1))
# 53. Using amin function



x = np.arange(10).reshape((2, 5))

print('x:')

print(x)



print('\nnp.amin(x, 1) : ')

print(np.amin(x, 1))
# 54. Using amax function



x = np.arange(10).reshape((2, 5))

print('x:')

print(x)



print('\nnp.amax(x, 1) : ')

print(np.amax(x, 1))
# 55. Get min of axis = 0



x = np.arange(6).reshape((2, 3))

print('x:')

print(x)



print('\nx.min(axis = 0) : ')

print(x.min(axis = 0))
# 56. Calculate 90th percentile of an axis



x = np.arange(6).reshape((2, 3))

print('x:')

print(x)



print('\nnp.percentile(x, 90, 0): ')

print(np.percentile(x, 90, 0))
# 57. Find median



x = np.arange(6).reshape((2, 3))

print('x:')

print(x)



print('\nnp.median(x): ')

print(np.median(x))
# 58. Covariance matrix



x = np.array([0, 1, 2])

y = np.array([7, 8, 9])



print('x:')

print(x)

print('\ny:')

print(y)



print('\nnp.cov(x, y): ')

print(np.cov(x, y))



# note: -1 means perfectly in opposite directions
# 59. Pearson product-moment correlation



x = np.array([0, 1, 3])

y = np.array([2, 4, 5])

print('x:')

print(x)

print('\ny:')

print(y)



print('\nnp.corrcoef(x, y): ')

print(np.corrcoef(x, y))
# 60. Cross correlation



x = np.array([0, 1, 3])

y = np.array([2, 4, 5])

print('x:')

print(x)

print('\ny:')

print(y)



print('\nnp.correlate(x, y): ')

print(np.correlate(x, y))
# 61. Count the number of occurrences



x = np.array([1, 2, 2, 1, 1, 1, 5, 5])



print('x:')

print(x)



print('\nnp.bincount(x): ')

print(np.bincount(x))



# how to interpret



#    starts from 0

#    ends with 5

    

#    0 appears 0 times

#    1 appears 4 times

#    ...
# 62. Create random int numpy array with specific shape



x = np.random.randint(10, 20, (4, 2))



print('np.random.randint(10, 20, (4, 2)):\n')

print(x)
# 63. Create 4 different integers from 0, 40. 



x = np.random.choice(40, 4, replace = False)



print('np.random.choice(40, 4, replace = False):\n')

print(x)
# 64. Create 4 different integers from 0, 4. (It will throw error as we can't get 5 unique integers from 4)



try:

    x = np.random.choice(4, 5, replace = False)

    

    print('np.random.choice(4, 5, replace = False):\n')

    print(x)

except Exception as err:

    print('Error : ', err)



# It is expected to throw an error
# 65. Shuffle



x = np.arange(10)



print('x Before shuffling:')

print(x)



print('\nx After shuffling:')

np.random.shuffle(x)

print(x)
# 66. Shuffling by permutation



x = np.arange(10)



print('x Before shuffling (using permutation):')

print(x)



print('\nx After shuffling (using permutation):')

print(np.random.permutation(10))
# 67. Seed for random

np.random.seed(12)



a = np.random.rand(4)

print('Random array after first seed:')

print(a)



np.random.seed(12)

b = np.random.rand(4)

print('\nRandom array after second seed:')

print(b)



c = np.random.rand(4)

print('\nRandom array after no seed:')

print(c)
# 68. Get unique elements



x = np.array([1, 2, 6, 4, 2, 3, 2])



print('x:')

print(x)



out, indices = np.unique(x, return_inverse=True)



print('\nunique elements of x:')

print(out)



unique, counts = np.unique(x, return_counts=True)

unique_dict = dict(zip(unique, counts))



print('\nunique elements and count:')

print(unique_dict)



import collections

col_counter = collections.Counter(x)

print('\nunique elements and count by collections.counter:')

print(col_counter)
# 69. Create a boolean array with a shape of x



x = np.array([0, 1, 2, 5, 0])

y = np.array([0, 1])



print('x:')

print(x)



print('\ny:')

print(y)



print('\nnp.in1d(x, y):')

print(np.in1d(x, y))
# 70. Find unique intersection



x = np.array([0, 1, 2, 5, 0])

y = np.array([0, 1, 4])



print('x:')

print(x)



print('\ny:')

print(y)



print('\nnp.intersect1d(x, y):')

print(np.intersect1d(x, y))
# 71. Find the diff



x = np.array([0, 1, 2, 5, 0])

y = np.array([0, 1, 4])



print('x:')

print(x)



print('\ny:')

print(y)



print('\nnp.setdiff1d(x, y):')

print(np.setdiff1d(x, y))
# 72. Trying to inverse a singular matrix



b = np.array([[2,3],[4,6]])



try:

    np.linalg.inv(b)

except Exception as err:

    print('Error : ', err)

    

# Note: Singular Matrix can't be inversed
# 73. Find the union



x = np.array([0, 1, 2, 5, 0])

y = np.array([0, 1, 4])



print('x:')

print(x)



print('\ny:')

print(y)



z = np.union1d(x, y)

print('\nnp.union1d(x, y):')

print(z)
# 74. Element Sum



x = np.array([0, 0, 1, 2, 1, 1, 0, 0, 0])

print('Original Array:')

print(x)



num_zeros = (x == 0).sum()

num_ones = (x == 1).sum()



print('\nzeros sum:')

print(num_zeros)



print('\nones sum:')

print(num_ones)



print('\nones sum by using list:')

x_list = list(x)

print(x_list.count(1))
# 75. Pretty print - suppress the scientific notation



x = np.array([1.2e-10, 2.1, 23])

print('Before:')

print(x)



np.set_printoptions(suppress = True)

print('\nAfter suppressing the scientific notation:')

print(x)



# reset

np.set_printoptions(suppress = False)
# 76. Print with decimal format



x = np.array([1.24500, 1.0000])

print('Before:')

print(x)



np.set_printoptions(formatter={'float': '{: 0.2f}'.format})

print('\nAfter formattting the float:')

print(x)



# Not sure how to reset it
# 77. Convert flot array to int array



x = np.array([[2.0, 8.8], [2.3, 1.9]])

print('Before:')

print(x)



y = x.astype(int)

print('\nAfter changing the datatype:')

print(y)



z = np.int_(x)

print('\nAfter changing the datatype by using int_:')

print(z)
# 78. array vs asarray



x = np.matrix(np.ones((3,3)))

print('Before:')

print(x)





np.array(x)[2] = 2

print('\nAfter changing elements by array:')

print(x)



np.asarray(x)[2] = 2

print('\nAfter changing elements by asarray:')

print(x)
# 79. Numpy Datetime



x = np.datetime64('2020-09-30')

print('Sample numpy datetime:')

print(x)

# print(type(x))



x1 = np.datetime64('2020-09', 'D')

print('\nNumpy Datetime with forcing D option:')

print(x1)

# print(type(x1))
# 80. Numpy Datetime with arange



x = np.arange('2020-02', '2020-06', dtype = 'datetime64[M]')

print('\nNumpy Datetime range with forcing month:')

print(x)





y = np.arange('2020-02', '2020-06', dtype = 'datetime64[D]')

print('\nNumpy Datetime range with forcing date:')

print(y)
# 81. Compare Numpy datetime



print("np.datetime64('2020') == np.datetime64('2020-01-01'):")

print(np.datetime64('2020') == np.datetime64('2020-01-01'))



print("\nnp.datetime64('2020-01') == np.datetime64('2020-01-02'):")

print(np.datetime64('2020-01') == np.datetime64('2020-01-02'))
# 82. Numpy 2D array flat list



import numpy as np

from itertools import chain



a = np.arange(12).reshape(2, 3, 2)

print('Before:')

print(a)



b = a.tolist()

print('\nAfter tolist:')

print(b)



c = list(chain.from_iterable(a))

print('\nAfter flattening the array to list:')

print(c)
# 83. Numpy 2D array to flat list



a = np.arange(6).reshape(2, 3)

print('Before:')

print(a)



b = list(a.flatten())

print('\nAfter flattening the array to list:')

print(b)
# 84. Numpy array to Pandas Dataframe



x = np.array([[90, 98], [92, 99]])

print('Numpy Array:')

print(x)



df = pd.DataFrame({'Maths': x[:, 0], 'Science': x[:, 1]})

print('\nDataframe from Numpy Array:')

print(df)
# 85. Numpy to Dataframe by using from_records



x = np.arange(6).reshape(2, -1) # -1 is used for lazy option

print('Numpy Array:')

print(x)



df = pd.DataFrame.from_records(x)

print('\nDataframe from Numpy Array by using from_records:')

print(df)
# 86. Append with hstack



a = np.arange(6).reshape(2, -1)

print('Numpy Array:')

print(a)



b = np.hstack((a, np.zeros((a.shape[0], 1), dtype = a.dtype)))

print('\nAfter appending with hstack:')

print(b)
# 87. Ravel vs. Flatten



a = np.array([[1,2],[3,4]])

print('Original numpy array:')

print(a)



ravel_a = np.ravel(a)

flatten_a = np.ndarray.flatten(a)  



print('\nravel a:')

print(ravel_a)



print('\nflatten a:')

print(flatten_a)



print('\nravel a base:')

print(ravel_a.base)



print('\nflatten a base:')

print(flatten_a.base) # since it is a copy, you would see None as base



# Note: if view, you will see the original array in .base

# If is a copy, you will see None in .base
# 88. Ravel and Flatten with assignment



a = np.array([[1,2],[3,4]])

print('Original numpy array, a:')

print(a)



ravel_a = np.ravel(a)

flatten_a = np.ndarray.flatten(a)  



print('\nravel a:')

print(ravel_a)



print('\nflatten a:')

print(flatten_a)



ravel_a[0] = 100

print('\nAfter changing ravel, a:')

print(a)



flatten_a[1] = 200

print('\nAfter changing flatten, a:')

print(a)



print('\nravel a base:')

print(ravel_a.base)



print('\nflatten a base:')

print(flatten_a.base) # since it is a copy, you would see None as base



# Note: if view, you will see the original array in .base

# If is a copy, you will see None in .base
# 89. Numpy Concatenate



a = np.arange(4).reshape(2, -1)

b = np.array([[5, 6]])



print('a:')

print(a)



print('\nb:')

print(b)



# appending as a row

c = np.concatenate((a, b), axis = 0)

print('\nAfter appending as a row:')

print(c)





# appending as a colum

d = np.concatenate((a, b.T), axis = 1)

print('\nAfter appending as a colum:')

print(d)





e = np.concatenate((a, b), axis = None)

print('\nAfter flatteeing and appending:')

print(e)
# 90. Reverse 1D array



x = np.arange(4)

print('Before:')

print(x)



y = x[::-1]

print('\nAfter reversing:')

print(y)
# 91. Reverse 2D Array



x = np.arange(8).reshape(2, -1)

print('Before:')

print(x)



y = x[::-1]

print('\nAfter reversing:')

print(y)
# 92. Reverse 2D array with elements



x = np.arange(8).reshape(2, -1)

print('Before:')

print(x)



y = np.flip(x)

print('\nAfter reversing with flip:')

print(y)



y = np.flipud(x)

print('\nAfter reversing with flipud:')

print(y)



y = np.fliplr(x)

print('\nAfter reversing with fliplr:')

print(y)
# 93. Image as an array

from PIL import Image



image_np = np.asarray(Image.open('/kaggle/input/numpy-cheatsheet/cn_tower.jpg'))

print(type(image_np))
# 94. Array with prefilled values



a = np.full((2, 4), 23)

print('Array with prefilled values:')

print(a)



b = np.empty((2, 4), dtype = int)

b.fill(23)

print('\nArray with prefilled values by using empty and fill:')

print(b)
# 95. Delete specific indices



a = np.array([1, 3, 5, 4, 7])

print('Before:')

print(a)



indices = [2, 3]

b = np.delete(a, indices)



print('\nAfter Deleting specific indices:')

print(b)
# 96. Deleting specific elements



a = np.array([1, 4, 5, 4])

print('Before:')

print(a)



b = np.array([3, 4])

c = np.setdiff1d(a, b)



print('\nAfter Deleting specific elements:')

print(c)
# 97. Boolean Numpy Array



a = np.ones((2, 2), dtype = bool)

print('True Boolean Array:')

print(a)



b = np.zeros((2, 3), dtype = bool)

print('\nFalse Boolean Array:')

print(b)



c = np.full((2, 4), True)

print('\nTrue Boolean Array by using .full:')

print(c)
# 98. Array of NaN



a = np.empty((3, 2,))

a[:] = np.nan

print('nan using empty:')

print(a)



b = np.full([3, 2], np.nan)

print('\nnan using full:')

print(b)
# 99. Replace values with specific condition



import numpy as np

a = np.random.rand(2, 4)

print('Before:')

print(a)



a[a > 0.5] = 0.5

print('\nAfter updating a[a > 0.5] = 0.5:')

print(a)
# 100. Selecting specific columns



x = np.arange(12).reshape(3, 4)

print('Before:')

print(x)



y = x[:, [1, 3]]

print('\nAfter selecting specific columns:')

print(y)
# 101. Vectorize



aa = np.array([[1,2,3,4], [2,3,4,5], [5,6,7,8], [9,10,11,12]])

bb = np.array([[100,200,300,400], [100,200,300,400], [100,200,300,400], [100,200,300,400]])



def vec2(a, b):

    return a + b



func2 = np.vectorize(vec2)

print(func2(bb[:,1], aa[:,1]))
# 102. If condition on Numpy array on the fly



a = np.arange(6).view([('b', np.int),('c', np.int)])



print(a)



print(a['b'] < 4)



print((a['b'] < 4).sum())
# 103. If condition and sum on Numpy colum



a = np.random.randint(0, 10, 20).reshape(4, 5)



print('Original Array:')

print(a)



# How many elements in column 3 are greater than 2?

b = a[:, 1] > 2



print('\nPrinting elements in colmns which are greater than 2: ')

print(b)



print('\nPrinting elements\' sum in colmns which are greater than 2: ')

print(b.sum())
# 104. Using Vectorize function on Numpy



def get_max(a, b):

    

    if (a > b): 

        return a

    

    return b

    

b_vectfunc = np.vectorize(get_max)

x = [[1, 2, 3], [4, 7, 2]]

y = [7, 4, 5]



result = b_vectfunc(x, y)



print(result)



# Note: It will compare the array element wise and get the max value
# 105. Apply Along Axis on Numpy Matrix



def get_avg(a):

    return (a[0] + a[-1]) / 2



a = np.random.randint(1, 10, 9).reshape(3, 3)



print('Original array:')

print(a)



b = np.apply_along_axis(get_avg, 0, a)



print('\nAfter averaging the first and last element:')

print(b)
# 106. Sort the arry by using 



a = np.random.randint(1, 10, 9).reshape(3, 3)



print('Original Array :')

print(a)



b = np.apply_along_axis(sorted, 0, a)



print('\nAfter sorting :')

print(b)
# 107. Numpy Roll



import numpy as np



x = np.arange(1, 7).reshape(2, 3)



print(x)



y = np.roll(x, 1, axis = 1)



print('\nAfter:')

print(y)
# 108. Roll elements



x = np.arange(10)

y = np.roll(x, 2)



print(x)



print(y)
# 109. Array Append on Axis 0 and Axis 1



a = np.array([1, 2])

b = np.array([3, 4])



print('Array a:')

print(a)



print('\nArray b:')

print(b)



c = np.append([a], [b], axis = 0)

print('\nArray c : append a and b on axis 0:')

print(c)



d = np.append([a], [b], axis = 1)

print('\nArray d : append a and b on axis 1:')

print(d)
# 110. Rearrange array with specified index



a = np.array([11, 22, 33, 44, 55])

print('Before:')

print(a)



idx = [4, 2, 0, 1, 3]



b = a[idx]



print('\nAfter:')

print(b)
# 111. Serialize array with pickle



a = np.array([10, 20])

print('Before:')

print(a)



import pickle



b = pickle.dumps(a, protocol = 4)



print('\nSerialized Numpy Array:')

print(b)



c = pickle.loads(b)



print('\nDe-serialized Numpy Array:')

print(c)



# Check protocols here: https://docs.python.org/3/library/pickle.html
# 112. Transpose



a = np.arange(6).reshape(2, -1) 



print('Before:')

print(a)



b = np.transpose(a)

print('\nAfter:')

print(b)
# 113. Continguous Array



a = np.ascontiguousarray(np.random.randint(0, 10, 5))

print(a)

print(a.flags)



b = np.array(np.random.randint(0, 10, 5))

print(b)

print(b.flags)



# Note: Not sure where we can use this option. I will do some research and update things here
# 114. Numpy Reshape with Order



import numpy as np



X = np.arange(12).reshape(6, 2)

print('Original:')

print(X)



Y = X.reshape(3, 2, 2, order = 'F')

print('\nReshape with Order F:')

print(Y)



# Options: 

# 'C' - # C-like index ordering

# 'F' - # Fortran-like index ordering

# 'A' - # Mixed Fortran or C (check docucmnetation for more)

# 'K' - ?
# 115. Element-wise math



a = np.arange(6).reshape(3, 2)

print('Before:')

print(a)



# b = x^2 + y^2

b = (np.array(a) ** 2).sum(-1)

print('\nAfter:')

print(b)
# 116. Where with multiple condition



dt = 1.0

a = np.arange(0.0, 5.0, dt)

print('Original:')

print(a)



# if a is less than 3 and greater than 0, multiply by 2 or else multiply by 4

b = np.where((a >= 0) & (a < 3), 2 * dt, 4 * dt)

print('\nAfter:')

print(b)
# 117. Subtract Outer



a, b = [2,7,8], [1,9]



c = np.abs(np.subtract.outer(a, b))

print('Original:')

print(c)



d = c.ravel()

print('\nAfter ravel:')

print(d)
# 118. Multiply Outer



a = np.array([1, 2, 3])

b = np.array([4, 5, 6])



c = np.multiply.outer(a, b)



print(c)



d = c.ravel()

print(d)
# 119. Immutable Array



a = np.arange(6)

a.flags.writeable = False



print('Before:')

print(a)



a[0] = 1 # this will throw ValueError: assignment destination is read-only
# 120. Float anomalies in Numpy Arange



for i in np.arange(0.0, 2.1, 0.1):

    print(i)

    

# source: 

# https://docs.python.org/3/tutorial/floatingpoint.html

# https://stackoverflow.com/questions/63824157/while-using-np-arange-it-was-incrementing-with-wrong-step-size
# 121. Complex number and absolute value



x = np.array(1.1 - 0.2j)

print('Original:')

print(x)



print('\nDatatype:')

print(x.dtype)



y = np.abs(x)

print('\nAbsolute Value:')

print(y)
# 122. Set Error in Numpy



np.seterr(all = "ignore")

a = np.mean([])  # this will be okay as we set the `ignore` option

print('Mean of empty array:')

print(a)



print('\nAfter setting error:')

np.seterr(all = "raise")

try:

    a = np.mean([])

except FloatingPointError as err:

    print('FloatingPointError: ', err)

    

# other options: 'ignore’, ‘warn’, ‘raise’, ‘call’, ‘print’, ‘log’
# 123. Convert List of List String to Numpy Array



import re



a_list = '''

[[25  3  2]

 [ 1 21  0]

 [ 1  0  0]]

'''



print('Original String:')

print(a_list)



rows = []



for line in filter(len, map(str.strip, a_list.split("\n"))):

    rows.append([ int(v) for v in re.findall(r"\b[0-9]+\b", line) ])



np_rows = np.array(rows)

print('\nAfter converting string to Numpy Array:')

print(np_rows)



# do this opeartion just to verify

np_rows += 2



print('\nAfter converting string to Numpy Array and add 2 in all elements:')

print(np_rows)



print('\nDatatype:')

print(type(np_rows))
# 124. Resize a list with rounds



a_list = [10, 20, 30]

result = np.resize(a_list, 12)



print(result)
# 125. Print columns by index



x = np.array([[1,10,11],[12,0,3]])

print('Original:')

print(x)



print('\nPrinting Cols with Index:')

for i in range(3):

    print(i, x[:,i])
# 126. Initalize array with tuples



a = np.full((3,2), np.nan, dtype='f,f')

print('Original:')

print(a)





a[0] = (12, 2)

print('\nAfter setting up a tuple:')

print(a)
# 127. Rearrange with permutation



a = np.array([[10, 20, 30],

                  [100, 200, 300]])



print('Original:')

print(a)



permutation = [1, 0]

b = a[permutation]

print('\nAfer:')

print(b)
# 128. Squeeze 1



x = np.array([[[0, 2], [1, 2], [2, 3]]])

print('Original:')

print(x)

print('\nShape:', x.shape)



y = np.squeeze(x)

print('\nAfter Squeezing:')

print(y)

print('\nShape:', y.shape)
# 129. Squeeze 2



x = np.array([[[0], [1], [2]]])

print('Original:')

print(x)

print('\nShape:', x.shape)



y = np.squeeze(x)

print('\nAfter Squeezing:')

print(y)

print('\nShape:', y.shape)
# 130. Convert Array indices to One Hot Encoded



a = np.array([1, 2, 4, 0])

print('Original:')

print(a)



b = np.zeros((a.size, a.max() + 1))

b[np.arange(a.size), a] = 1



print('\nAfter:')

print(b)
# 131. Business Day Count



a = np.busday_count('2020-01', '2021-01')



print(a)
# 132. Multinomial 1



size   = 11

levels = 6

a = np.random.multinomial(size, np.ones(levels)/levels, size=1)



print(a)



# 133. Multinomial 2



b = np.random.multinomial(20, [1/6.]*6, size=1)

print(b)



# How to interpret?

# Assuue, you throw dice 20 times, if you get a result [[2 3 1 5 3 6]], it means

# You got 1 for 2 times

# You got 2 for 3 times
# 134. Random Generator - Draw samples from Beta Distribution



from numpy.random import default_rng



rng = default_rng()



a = np.array([1, 2, 3])

b = np.array([7, 8, 9])



rng.beta(a + 10, b + 10)



# Doc: https://numpy.org/doc/stable/reference/random/index.html

# https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.beta.html#numpy.random.Generator.beta
# 135. Get only unique



import numpy as np



a = np.array(['a', 'b', 'b', 'c', 'c', 'd', 'd'])

print('Original:')

print(a)



b = np.unique(a)

print('\nUnique:')

print(b)
# 136. Find indices with SearchSorted



import numpy as np



a = np.searchsorted([1,2,3,4,5], 3)

print('np.searchsorted([1,2,3,4,5], 3):')

print(a)





b = np.searchsorted([1,2,3,4,5], [7, 2, 3])

print('\nnp.searchsorted([1,2,3,4,5], [7, 2, 3]):')

print(b)
# 137. Loop list by using itertools.cycle



import itertools



a_list = ["CA", "ON", "NY"]

print('Original:')

print(a_list)



rounds = 5

ic = itertools.cycle(a_list)



a = [next(ic) for _ in range(rounds)]

print('\nAfter looping 5 rounds:')

print(a)



# You can do the same in Numpy resize to get Numpy Array

b = np.resize(a_list, rounds)

print('\nAfter looping 5 rounds by using Numpy Resize:')

print(b)
# 138. Applying Triu function



import numpy as np



a = np.matrix([[1,2,3],[4,5,6],[7,8,9]])

print("Original:")

print(a)



print("\ntriu() without any parameter:")

print(np.triu(a))



print("\nAbove 1st diagonal zeroed:")

print(np.triu(a, 1))



print("\nBelow 1st diagonal zeroed:")

print(np.triu(a, -1))
# 139. Unpack list elements



a_list = [[1, 2, 3], [4, 5, 6]]

b_list = np.concatenate(a_list).ravel()



print(b_list)
# 140. Partitioning



a = np.array([8, 4, 2, 3, 1, 7, 10, 12])

print('Original:')

print(a)



b = np.partition(a, 5)

print('\nAfter partitioning:')

print(b)
# 141. Expand Dimension



import numpy as np

a = np.array([2, 4])

print('Original:')

print(a)

print('\nShape:')

print(a.shape)



b = np.expand_dims(a, axis = 0)

print('\nAfter expanding dim:')

print(b)

print('\nShape:')

print(b.shape)
# 142. Gradient sample



a = np.array([1, 3, 4], dtype = float)

print('Original:')

print(a)



b = np.gradient(a)

print('\nAfter np.gradient(a):')

print(b)



c = np.gradient(a, 2)

print('\nAfter np.gradient(a, 2):')

print(c)
# 143. 



a = np.array([[10, 20], [30, 40]])

print('Original:')

print(a)



b = a.flatten()

print('\nAfter flatten:')

print(b)



c = a.flatten('F')

print('\nAfter flatten with F:')

print(c)
# 144. Check atlas



import numpy.distutils.system_info as sysinfo

sysinfo.get_info('atlas')
# 145. Check Numpy Configuration



np.show_config()