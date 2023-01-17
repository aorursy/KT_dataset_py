# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # numerical python library

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # plotting



# We can also import specific functions that we might think are useful

from numpy.linalg import solve # function for solving set of linear equations

                                # ie equations of the form Ax = b
# Let's set up some numpy arrays



A = np.array([[1, 2], [3,4]])



print(A)



b = np.array([5, 6])



x = solve(A,b)



print(x)



# We can check solution (always check!) by using 



print("b = ", np.dot(A,x))