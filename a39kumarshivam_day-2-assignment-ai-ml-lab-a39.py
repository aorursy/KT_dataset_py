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
# 1.Create 1D,2D and boolean array using numpy.

#1D array

import numpy as np

a = np.array([1,1,2,3,5,8,13,21,34,55,79])

print(a)
#2D array

import numpy as np

b = [2,3,5,7],[73,37,19,17],[101,111,153,59]

c = np.array(b)

print(c)

#boolean array

import numpy as np

d = np.array([1,1,.5,0,0.66], dtype=bool)

print(d)
# 2. Extract the odd numbers from a 2D array using numpy package.

import numpy as np

b = np.array([[2,3,5], [1,6,11]], np.int32)

c = (b[b%2!=0])

print(c)
# 3. How to replace items that satisfy a condition with another value in numpy array?

# this is done by following syntax

# Arraynew = (<condition of the array>).<function>
# 4. How to get the common items between two python numpy arrays?

import numpy as np

a1 = np.array([2,3,5,7,11])

print(a1)

a2 = [1,2,3,4,5,6,7,8,9]

print(a2)

print("common values between two arrays are")

print(np.intersect1d(a1,a2))
# 5. How to remove from one array those items that exist in another?

import numpy as np

b1 = np.array([2,4,6,8,10,12])

print(a1)

b2 = [1,2,3,4,5,6,7,8,9,10]

print(a2)

for i in a1:

    if i in a2:

        a2.remove(i)

print("new array is",a2)        
# 6. How to get the position where elements of two arrays match?

import numpy as np

a1 = np.array([1,2,3,4,6,10])

a2 = np.array([10,9,5,4,2,3])

np.where(a1==a2)