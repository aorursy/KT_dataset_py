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
# Identity Matrix

# Since all Identity Matrices are square, the np.eye() function only takes a single integer as an argument

# 7 x 7 Identity matrix

X = np.eye(7)

print(X)
# Diagonal Matrix

# 8 x 8 diagonal matrix that contains the numbers 1,2,3,4,5,6,7 and 8 on its main diagonal

X = np.diag([1,2,3,4,5,6,7,8])

print(X)
# Slicing

# NumPy array with elements from 1 to 9 

x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]) 



# Index values can be negative. 

arr = x[np.array([1, 3, -3])] 

print("Elements are : \n",arr) 
# Stacking 

arrays = [np.random.randn(2, 3)for _ in range(8)]

a = np.stack(arrays, axis=0).shape

print(a)
# Broadcasting 

a = np.array([1,2,3,4]) 

b = np.array([10,20,30,40]) 

c = a * b 

print(c)