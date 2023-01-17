import random

import numpy as np

from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt

%matplotlib inline
# Function for sigmoid

def sigmoid(a):

    return ((1.0)/(1 + np.exp(-a)))



# Gradient descent main Function

def grad_desc(X, y, alpha, num_iter, theta, m):

    

    X_trans = X.T

    

    for i in range(0, num_iter):

        prediction = sigmoid(np.dot(X, theta))

        error = sigmoid(prediction) - y

        cost = (np.dot(-y, np.log(prediction)) - np.dot((1 - y), np.log(1 - prediction))) / m 

#         Calculation for gradient descent starts now

        gradient = np.dot(X_trans, error) / m

        theta = theta - alpha * gradient

    

    return theta, prediction



# DATA PLOTTING STARTS NOW



X = np.zeros((100,2))

X[:,0] = np.ones(100)

X[:,1] = np.linspace(-3, 3, 100)  #it takes values in increasing order from -3 to 3 thats why we use linspace

z = (7.2)*(X[:,1])

noise = 0.3*np.random.randn(100)

y = 1/(1+np.exp(-z))+noise

# print(y)

plt.scatter(X[:,1],y)





y = (y >= 0.5) # returns boolean value to y (false ot true)

# y1 = y

y = y*1



m, n = np.shape(X)

theta = np.ones(n)

num_iter = 100 # number of iteration u want to perform feel free to change it as much as u want

alpha = 0.1 #Alpha or so called learning factor

theta, h = grad_desc(X, y, alpha, num_iter, theta, m)

print("\nValues of THETA after "+str(num_iter)+" Iteration is = \n")

print(theta)



plt.scatter(X[:,1],y)

plt.xlabel('x axis')

plt.ylabel('y axis')

plt.scatter(X[:,1],y)

plt.plot(X[:,1],h)
# print(y1)

# print(z)