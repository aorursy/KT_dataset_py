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

#
#1d array

import numpy as np



#create numpy array

a = np.array([5, 8, 12])

print(a)
#2d array

import numpy as np

x = np.array([[2, 4, 6], [6, 8, 10]], np.int32)

print(type(x))	

print(x.shape)

print(x.dtype)
#boolean

import numpy as np



bool_arr = np.array([1, 0.5, 0, None, 'a', '', True, False], dtype=bool)

print(bool_arr)

#odd no. 

a=np.array([1,2,3,4,5,6])

power = 2

answer = (a[a%2==1])**power

print (answer)
#match condn. 

import numpy as np

a = np.random.randint(0, 5, size=(5, 4))

b = np.where(a<3,0,1)

print('a:',a)

print()

print('b:',b)
#common items



import numpy as np

array1 = np.array([0, 10, 20, 40, 60])

print("Array1: ",array1)

array2 = [10, 30, 40]

print("Array2: ",array2)

print("Common values between two arrays:")

print(np.intersect1d(array1, array2))

import numpy as np



A = np.array([[1, 1, 1,], [1, 1, 2], [1, 1, 3], [1, 1, 4]])

B = np.array([[0, 0, 0], [1, 0, 2], [1, 0, 3], [1, 0, 4], [1, 1, 0], [1, 1, 1], [1, 1, 4]])

A_rows = A.view([('', A.dtype)] * A.shape[1])

B_rows = B.view([('', B.dtype)] * B.shape[1])



diff_array = np.setdiff1d(A_rows, B_rows).view(A.dtype).reshape(-1, A.shape[1])

diff_array
import numpy



a = numpy.array([0, 1, 2, 3, 4, 5, 6])

b = numpy.array([6, 5, 4, 3, 2, 1, 6])

numpy.where(a==b)