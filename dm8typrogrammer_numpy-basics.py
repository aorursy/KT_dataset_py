# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
n = np.array([1, 2, 3])

print(n, type(n), n.shape, n.dtype)
n= np.arange(1,11)  # range: [stat, end)

print(n, type(n), n.shape, n.dtype)
n2 = n.reshape(2, 5)

print(n2, type(n2), n2.shape, n.dtype) #  


n2 = np.array([[1, 2, 3], [2, 3, 4], [2, 3, 4]])

print(n2, type(n2), n2.shape, n.dtype )
array1 = np.array([[1, 2], [4, 5]])

array2 = np.array([[3],[6]])

mergedArray = np.hstack((array1, array2)) # note double brackets

print(mergedArray)
array1 = np.array([[1, 2]])

array2 = np.array([[3, 4]])

mergedArray = np.vstack((array1, array2)) # note double brackets

print(mergedArray)
array = np.array([[1, 2],[3, 4]])

print('before Transport:')

print(array)

print('after Transport:')

print(array.T)
# square of each number



# First define a operation spec: 

operation = np.vectorize(lambda x: x ** 2)



array = np.array([1, 2, 3])

operation(array)