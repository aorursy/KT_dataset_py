import numpy as np

import pandas as pd 

import os

print(os.listdir("../input"))
from scipy.optimize import linprog

c = [-3, -2]

A = [[1,1],[2,1],[1,0]]

b = [80,100,40]

x0_bounds = (0, None)

x1_bounds = (0, None)

result = linprog(c, A_ub=A, b_ub=b, bounds=(x0_bounds,x1_bounds),method = 'simplex' )

print(result)