import numpy as np



def func(x):

    return x**2+5



x = np.array([-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6])

y = func(x)



print(x)

print(y)
%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns; sns.set()
plt.plot(x,y,marker='o',color='b',linestyle='-');

plt.plot(x, y, 'o', color='r');
import sympy as sp



x = sp.Symbol('x')



f = x**2 + 5

f_prime = f.diff(x)



print(f_prime)
gamma = 0.001  

x = 6

iterations = 5000

    

for i in range(0,iterations):

    x_gradient = 2*x

    x1 = x - gamma * x_gradient

    x = x1

        

print("iterations =",iterations,"\nx = ",x)