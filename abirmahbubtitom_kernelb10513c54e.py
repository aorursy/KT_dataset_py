from scipy.optimize import linprog

c = [-2,-12] 

A = [[3,7],[0,5],[-1,0]]

b = [79,42,-3]

x0_bounds = (0, None)

x1_bounds = (0, None)

result = linprog(c, A_ub=A, b_ub=b, bounds=(x0_bounds,x1_bounds),method = 'simplex' )

print(result)