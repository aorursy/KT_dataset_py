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
# Create a 1D ndarray that contains only integers

x = np.array([1, 2, 3, 4, 5])

print('x = ', x) # x = [1 2 3 4 5]

print('x has dimensions:', x.shape) # x has dimensions: (5,)

print('The elements in x are of type:', x.dtype) # The elements in x are of type: int64



# Create a rank 2 ndarray that only contains integers

Y = np.array([[1,2,3],[4,5,6],[7,8,9], [10,11,12]])

print('Y has dimensions:', Y.shape) # Y has dimensions: (4, 3)

print('Y has a total of', Y.size, 'elements') # Y has a total of 12 elements

print('Y is an object of type:', type(Y)) # Y is an object of type: class 'numpy.ndarray'

print('The elements in Y are of type:', Y.dtype) # The elements in Y are of type: int64
# Specify the dtype when creating the ndarray

x = np.array([1.5, 12.2, 35.7, 46.0, 59.9], dtype = np.int64)

print(x)
# Create ndarray using built-in functions

# 5 x 4 ndarray full of zeros

# np.zeros(shape)

X = np.zeros((5,4))

print(X)
# a 4 x 2 ndarray full of ones

# np.ones(shape)

X = np.ones((4,2))

print(X)
# 2 x 3 ndarray full of fives

# np.full(shape, constant value)

X = np.full((2,3), 5)

print(X)