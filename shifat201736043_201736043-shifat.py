c = [-3,-2]

A = [1,1],[2,1],[1,0]

b = [80,100,40]

x0_bounds = (0, None)

x1_bounds = (0, None) 

from scipy.optimize import linprog

solution = linprog(c, A, b, bounds = (x0_bounds , x1_bounds) , method = 'simplex')

print (solution)