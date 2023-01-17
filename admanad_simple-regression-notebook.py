import pandas as pd

import numpy as np

import os

data = pd.read_csv("../input/ex1data.txt", header=None)

data.columns = ["pop_10k","profit_10k"]

data["b"] = 1
X = data[["b","pop_10k"]]

y = data["profit_10k"]
data.head()
import matplotlib.pyplot as plt



plt.plot(X["pop_10k"], y, 'ro')

plt.xlabel("population of city in 10000")

plt.ylabel("profit in 10000")

m = len(y)

plt.show()
X_ = X.values

y_ = y.values.reshape(m,1)
theta = np.zeros((2,1))
# Some gradient descent settings

iterations = 1500

alpha = 0.01
def computeCost(X, y, theta):

    #Initialize some useful values

    m = len(y)

    #You need to return the following variables correctly 

    mul=1/(2*len(y))

    ###### COMPLETE ###############

    mat= None

    ###### COMPLETE ###############

    p = (X@theta)

    J=mul*mat

    return J

    
J = computeCost(X_, y_, theta)

J
def gradientDescent(X, y, th, alpha, num_iters):

    # GRADIENTDESCENT Performs gradient descent to learn theta

    # theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 

    # taking num_iters gradient steps with learning rate alpha



    # Initialize some useful values

    m = len(y) # number of training examples

    J_history = []

    (m, n) = X.shape

    print(y.shape)

    for iter in range(num_iters):

        #====================== YOUR CODE HERE ======================

        #Instructions: Perform a single gradient step on the parameter vector

        #theta. 

        #

        #Hint: While debugging, it can be useful to print out the values

        #of the cost function (computeCost) and gradient here.

        #         

        ###### COMPLETE ###############  

        theta1 = None

        theta2 = None

        ###### COMPLETE ###############

        th[0,0] = theta1

        th[1,0] = theta2



        #============================================================

        #Save the cost J in every iteration



        J_history.append(computeCost(X, y, th))



    return (th, J_history)



(param, _) = gradientDescent(X_, y_, theta, alpha, iterations)

param
plt.plot(X["pop_10k"], y, 'ro')

plt.xlabel("population of city in 10000")

plt.ylabel("profit in 10000")

i = np.linspace(5,23,23)

j = param[0,0]+param[1,0]*i

plt.plot(i, j, '-b', label='y=theta0*x+theta1')

plt.legend(loc='upper left')

plt.title('Linear regression result')



m = len(y)

plt.show()