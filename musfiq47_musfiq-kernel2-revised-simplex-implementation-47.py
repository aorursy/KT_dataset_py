import numpy as np

import scipy as np
# Define the Linear Programming

c = [-2, -47] # Coefficient of Objective function (Minimization)

A = [[-1, 0], [0, 5], [3, 7]] # LHS of the constraints

b = [-3, 42, 79]  # RHS of the constraints

x0_bounds = (0, None)

x1_bounds = (0, None)



# Import the Optimization library

from scipy.optimize import linprog

# Solve the problem by Simplex method in Optimization

result = linprog(c, A_ub=A, b_ub=b,  bounds=(x0_bounds, x1_bounds), method='simplex', options={"disp": True})

print(result)