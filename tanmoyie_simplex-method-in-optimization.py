"""

Related YouTube Video: https://www.youtube.com/watch?v=e2lHyMl1IYY&index=2&list=PLHyZ7Tamw-fevmrx2V3U13hPDDlUSBbi7 

"""

# Define the Linear Programming

c = [-3, -5] # Coefficient of Objective function (Minimization)

A = [[1, 0], [0, 2], [3, 2]] # LHS of the constraints

b = [4, 12, 18]  # RHS of the constraints

x0_bounds = (0, None)

x1_bounds = (0, None)



# Import the Optimization library

from scipy.optimize import linprog

# Solve the problem by Simplex method in Optimization

result = linprog(c, A_ub=A, b_ub=b,  bounds=(x0_bounds, x1_bounds), method='simplex', options={"disp": True})

print(result)