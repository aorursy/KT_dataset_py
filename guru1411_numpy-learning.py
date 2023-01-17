# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
np.arange(0, 10)
np.arange(0, 20, 2)
np.zeros((5, 5))
np.ones((2, 4), dtype = float)
array = np.random.randint(0, 10, 5)
array.max()
array.mean()
array
array2 = np.random.randint(0, 100, 10)
array2
array2.reshape((2, 5))
array3 = np.arange(0, 16, dtype = float).reshape((4, 4))
array3
A = np.array([[2, 5], [3, 8]])
A
B = np.array([[3, 4], [2, 3]])
B
A + B
A * B
A
B
A.dot(B)
A.sum()
B.sum()
A.max()
B.max()
A
A.sum(axis=0)
A.sum(axis=1)
A.min(axis=0)
A.T
# Create array of 10 zeroes

# Create array of 10 ones

# Create array of integers from 10 to 50

# Create 3 * 4 matrix with values 0 to 12

# Interview question: Diff between numpy and list ( Why should we use numpy arange instead of python list?)
np.zeros((10, 10))
np.ones((10, 10))
np.arange(10, 50)
np.arange(0, 12).reshape((3 ,4))