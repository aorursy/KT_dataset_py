from scipy.optimize import linprog
c=[-2,-40]

A=[[-1,0],[0,5],[3,7]]

b=[-3,42,79]

x0_bounds = (0, None)

x1_bounds = (0, None)

result=linprog(c,A_ub=A,b_ub=b,bounds=(x0_bounds,x1_bounds),method='simplex')

print(result)