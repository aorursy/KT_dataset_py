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
#Question 1a

import numpy as np

#create 1d array

a = np.array([1,2,3,4,5])
print(a)
#Question 1b

import numpy as np

#create 2d array

a2 = np.array([[1,2,3,4,5],[6,7,8,9,10]], np.int32)
print(a2)
print(type(a2))
print(a2.shape)
print(a2.dtype)
#Question 1c

import numpy as np

#create boolean array

barr = np.array([1,0,5,0,None,'a','',True,False], dtype=bool)
print(barr)
#Question 2

import numpy as np

#extract odd numbers from 2d array

a2 = np.array([[1,2,3,4,5],[6,7,8,9,10]], np.int32)
ans = (a2[a2%2 != 0])
print(ans)
#Question 3

import numpy as np

#replace items that satisfy a condition with another value
#here, we replace all odd numbers with 0

a = np.random.randint(0,5,size=(4,4))
print(a)
a[a%2 != 0] = 0
print(a)
#Question 4

import numpy as np

#get common items between two python numpy arrays

a = np.array([i for i in range(5,100,5)])
b = np.array([i for i in range(8,100,8)])
print("Multiples of 5 under 100:",a)
print("Multiples of 8 under 100:",b)
print("Multiples of both 5 and 8 under 100:",np.intersect1d(a,b))
#Question 5

import numpy as np

#remove from one array those items that exist in another

a = [i for i in range(0,10)]
b = np.array([i for i in range(0,10,2)])
print(a)
print(b)
for i in a:
    if i in b:
        a.remove(i)
print(a)
#Question 6

import numpy as np

#get the positions where elements of two arrays match

a = np.array([0,1,2,3,4,5])
b = np.array([5,1,3,3,0,6])
np.where(a==b)