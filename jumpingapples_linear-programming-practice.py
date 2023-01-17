# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from scipy.optimize import linprog as lp
from numpy.linalg import solve as sv

# Vivek goes to the store to buy fruit
# He wants to spend as little money as possible
# He needs 100 fruit
# 1 apple costs $1, 2 orange costs $1.5, 1 pear costs $1.5
# Buying 1 apple takes 2 minutes, 2 oranges takes 5 minutes, 1 pear takes 1 minute
# We have a maximum of 150 minutes to buy fruit
A_eq = np.array([[1, 2, 1]]) # How much does each option contribute to our count of fruits?
b_eq = np.array([100]) # We want to buy 100 fruit

A_ub = np.array([[2, 5, 1]]) # How many minutes does each option take
b_ub = np.array([150]) # Maximum minutes

c = np.array([1, 1.5, 1.25]) # How much does each option contribute to what we are minimizing

res = lp(c, A_eq = A_eq, b_eq = b_eq, A_ub = A_ub, b_ub = b_ub, bounds = (0, None))
print("Optimal Value: ", res.fun, "\nX: ", res.x)
