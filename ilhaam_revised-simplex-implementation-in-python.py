# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np

import pandas as pd

import os

print(os.listdir("../input"))
from scipy.optimize import linprog

c = [-2,-24] 

A = [[3,7],[0,5],[-1,0]]

b = [79,42,-3]

x0_bounds = (0, None)

x1_bounds = (0, None)

result = linprog(c, A_ub=A, b_ub=b, bounds=(x0_bounds,x1_bounds),method = 'simplex' )

print(result)