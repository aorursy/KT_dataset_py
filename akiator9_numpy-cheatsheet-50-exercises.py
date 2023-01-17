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
# 10. Numpy array with strings and explicit dtype



x = np.array(['To'], dtype=str)

y = x.view('S1').reshape(x.size, -1)



print(y)
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
# 22. Trying to inverse a singular matrix



b = np.array([[2,3],[4,6]])



try:

    np.linalg.inv(b)

except Exception as err:

    print('Error : ', err)

    

# Note: Singular Matrix can't be inversed
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
# 42. 



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
# 45. Argsort on Numpy array



a = np.random.randint(0, 10, (3,3))

print('Before : ')

print(a)



print('\nAfter : ')

b = a[a[: ,2].argsort()]

print(b)
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