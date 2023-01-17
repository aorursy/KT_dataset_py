c=[-2,-36] #Coefficient of Objective function  #36 is the last 2 digits of my roll number

A=[-1,0] #LHS of the constraints

b=[-3,42,79] #RHS of the constraints

x0_bounds = (0, None)

x1_bounds = (0, None)

from scipy.optimize import linprog

result=linprog(c,A_ub=A,b_ub=b,bounds=(x0_bounds,x1_bounds),method='simplex')

print(result)