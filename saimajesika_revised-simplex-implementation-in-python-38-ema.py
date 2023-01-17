import numpy as np

import scipy as sp
c=[-2,-38]

A=[[1,0],[0,5],[3,7]]

b=[3,43,79]

x0_bounds=(0,None)

x1_bounds=(0,None)
from scipy.optimize import linprog

result=linprog(c,A_ub=A,b_ub=b,bounds=(x0_bounds),method='simplex')

print(result)