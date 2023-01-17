%matplotlib inline

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (20.0, 10.0)

from mpl_toolkits.mplot3d import Axes3D



data = pd.read_csv('../input/bike-share-daily-data/bike_sharing_daily.csv')

del data['dteday']

data = (data-data.mean())/data.std() # normalization



# print(data.shape)

data.head()
# Ploting the scores as scatter plot

atemp = data['atemp'].values          # values to be an array

humidity = data['hum'].values

windspeed = data['windspeed'].values

cnt = data['cnt'].values

fig = plt.figure()

ax = Axes3D(fig)

ax.scatter(atemp, humidity,windspeed, cnt, color='#ef1234')

plt.show()
m = len(atemp)

x0 = np.ones((m,1))

# X = np.array([x0, atemp, humidity,windspeed]).T

X = data[['atemp','hum','windspeed']].values 

X = np.concatenate((x0 , X), axis=1)

# Initial Coefficients

theta = np.zeros((4,1))

Y = cnt.reshape(len(cnt),1)

alpha = 0.0001

# First Method for mathematicians

from numpy.linalg import inv

thetas = np.matmul(np.matmul(inv(np.matmul(X.T,X)), X.T),Y)

thetas
def cost_function(X, Y, theta):

    m = len(Y)

    J = np.sum((X.dot(theta) - Y) ** 2)/(2 * m)

    return J
inital_cost = cost_function(X, Y, theta)

print(inital_cost)
# Second Method for machine learners

def gradient_descent(X, Y, theta, alpha, iterations):

    cost_history = [0] * iterations

    m = len(Y)

    

    for iteration in range(iterations):

        # Hypothesis Values

        h = X.dot(theta)

        # Difference b/w Hypothesis and Actual Y

        loss = h - Y

        # Gradient Calculation

        gradient = X.T.dot(loss) / m

        # Changing Values of theta using Gradient

        theta = theta - alpha * gradient

        # New Cost Value

        cost = cost_function(X, Y, theta)

        cost_history[iteration] = cost

        

    return theta, cost_history 
# 100000 Iterations

newTheta, cost_history  = gradient_descent(X, Y, theta, alpha, 50000)



# New Values of theta

print(newTheta)



# Final Cost of new theta

print(cost_history[-1])

plt.plot(cost_history)

plt.title('Cost Function Changine')

plt.ylabel('J Function')

plt.xlabel('Iterations')