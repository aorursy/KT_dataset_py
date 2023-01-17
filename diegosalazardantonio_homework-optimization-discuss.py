import matplotlib.pyplot as plt

import numpy as np

import cvxpy as cp



"""

Q1, Q2, Q3:

formatting the convex optimization problem:

Af = b

f_min <= f_i <= f_max

minimize z = ||f||_p

We could always use regressive approach to obtain the global optimum because of the nature of convex

BUT here we use the CVXPY library

"""

# Construct the problem.

n = 4

x=0

aux=np.zeros(n)

for i in range(-n // 2,0,1):

    aux[x] = i

    x+=1

for i in range(1,n // 2 + 1,1):

    aux[x] = i

    x+=1    

A = np.array([np.ones(n), aux])

b = np.array([10, 2])

# number of rotors







##############################################################################################

##############################################################################################

from scipy.optimize import minimize



#objetctive function

def objective(fi):

    return np.linalg.norm(fi,1)



#contraint

def constraint1(fi):

    return A.dot(fi.transpose())-b



con2 = {'type': 'eq', 'fun': constraint1}



# initial guesses

f0 = np.ones(n)



bon = (0,60000)

for i in range(0,n-1,1):

    bnds = np.vstack((bnds, bon)) 



solution = minimize(objective,f0, constraints=con2)

x_norm1 = solution.x



plt.bar(A[1], x_norm1,width=0.2)

plt.grid(True)


from scipy.optimize import least_squares



A1 = np.array([[1,1,1,1], [-2,-1,1,2]])

b1 = np.array([10, 2])



#objetctive function

def objective(fi):

    return (fi[0]+fi[3])+fi[1]+fi[2]



#contraint

def constraint1(fi):

    return A1.dot(fi.transpose())-b1



con21 = {'type': 'eq', 'fun': constraint1}



# initial guesses

f01 = np.ones(4)



bo = (0,60000)

bnds1 = (bo,bo,bo,bo)



solution = minimize(objective,f01,bounds=bnds1, constraints=con21)

x_least_squares = solution.x



plt.bar(A1[1], x_least_squares,width=0.2)

plt.grid(True)
#objetctive function

def objective(fi):

    return np.linalg.norm(fi,2)



#contraint

def constraint1(fi):

    return A.dot(fi.transpose())-b



con2 = {'type': 'eq', 'fun': constraint1}



# initial guesses

f0 = np.ones(n)



bon = (0,60000)

for i in range(0,n-1):

    bnds = np.vstack((bnds, bon)) 

    

solution = minimize(objective,f0, constraints=con2)

x_norm2 = solution.x



plt.bar(A[1], x_norm2,width=0.2)

plt.grid(True)
#objetctive function

def objective(fi):

    return np.linalg.norm(fi,np.inf)



solution = minimize(objective,f0, constraints=con2)

x_norm_infinit= solution.x



plt.bar(A[1], x_norm_infinit,width=0.2)

plt.grid(True)