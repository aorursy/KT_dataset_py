import numpy as np

import pandas as pd

import math 

import random as rn

import matplotlib.pyplot as plt
data=pd.read_csv('../input/iris-flower-dataset/IRIS.csv')
X=data.iloc[:,0]

Y=data.iloc[:,1]

plt.scatter(X,Y)

plt.show()
m=0

c=0

L=0.0001

epochs=1000

n=float(len(X))

for i in range(epochs): 

    Y_pred = m*X + c  # The current predicted value of Y

    D_m = (-2/n) * sum(X * (Y - Y_pred))  # Derivative wrt m

    D_c = (-2/n) * sum(Y - Y_pred)  # Derivative wrt c

    m = m - L * D_m  # Update m

    c = c - L * D_c  # Update c

    

print (m, c)
Y_pred = m*X + c



plt.scatter(X, Y) 

plt.plot([min(X), max(X)], [min(Y_pred), max(Y_pred)], color='red')  # regression line

plt.show()
df = pd.read_csv('../input/iris-flower-dataset/IRIS.csv')

df.head()
X = df.iloc[0:100, [1, 2]].values



y = df.iloc[0:100,4 ].values



y = np.where(y == 'Iris-setosa', 1, 0)
X_std = np.copy(X)



X_std[:,0] = (X_std[:,0] - X_std[:,0].mean()) / X_std[:,0].std()

X_std[:,1] = (X_std[:,1] - X_std[:,1].mean()) / X_std[:,1].std()
def sigmoid(X, theta):

    

    z = np.dot(X, theta[1:]) + theta[0]

    

    return 1.0 / ( 1.0 + np.exp(-z))
def lrCostFunction(y, hx):

  

    # compute cost for given theta parameters

    j = -y.dot(np.log(hx)) - ((1 - y).dot(np.log(1-hx)))

    

    return j
def lrGradient(X, y, theta, alpha, num_iter):

    # empty list to store the value of the cost function over number of iterations

    cost = []

    

    for i in range(num_iter):

        # call sigmoid function 

        hx = sigmoid(X, theta)

        # calculate error

        error = hx - y

        # calculate gradient

        grad = X.T.dot(error)

        # update values in theta

        theta[0] = theta[0] - alpha * error.sum()

        theta[1:] = theta[1:] - alpha * grad

        

        cost.append(lrCostFunction(y, hx))

        

    return cost
m, n = X.shape



theta = np.zeros(1+n)



alpha = 0.01

num_iter = 500



cost = lrGradient(X_std, y, theta, alpha, num_iter)
plt.plot(range(1, len(cost) + 1), cost)

plt.xlabel('Iterations')

plt.ylabel('Cost')

plt.title('Logistic Regression')
print ('\n Logisitc Regression bias(intercept) term :', theta[0])

print ('\n Logisitc Regression estimated coefficients :', theta[1:])