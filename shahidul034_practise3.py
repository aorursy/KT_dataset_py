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
# import the "pyplot" submodule from the "matplotlib" package with alias "plt"

import matplotlib.pyplot as plt
from numpy import array

numbers = array([3, 4, 20, 15, 7, 19, 0]) # works fine

#numbers = numpy.array([3, 4, 20, 15, 7, 19, 0]) # NameError: name 'numpy' is not defined

type(numbers) # numpy.ndarray
import numpy as np 

np.array([False, 42, "Data Science"])       # array(["False", "42", "Data Science"], dtype="<U12")

np.array([False, 42], dtype = int)          # array([ 0, 42])

np.array([False, 42, 53.99], dtype = float) # array([  0.  ,  42.  ,  53.99])



# Invalid converting

#np.array([False, 42, "Data Science"], dtype = float) # could not convert string to float: 'Data Science'
np.array([37, 48, 50]) + 1 # array([38, 49, 51])

np.array([20, 30, 40]) * 2 # array([40, 60, 80])

np.array([42, 10, 60]) / 2 # array([ 21.,   5.,  30.])



np.array([1, 2, 3]) * np.array([10, 20, 30]) # array([10, 40, 90])

np.array([1, 2, 3]) - np.array([10, 20, 30]) # array([ -9, -18, -27])
numbers = np.array([

    [1, 2, 3],

    [4, 5, 6],

    [7, 8, 9]

])



np.dot(numbers,numbers)