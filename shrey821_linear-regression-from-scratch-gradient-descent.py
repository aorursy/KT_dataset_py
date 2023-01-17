import numpy as np

import pandas as pd

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

import io

import base64
import matplotlib.pyplot as plt

import matplotlib.animation as animation

data = pd.read_csv('../input/train.csv')
x = data['GrLivArea']

y = data['SalePrice']

x = (x - x.mean()) / x.std()

x = np.c_[np.ones(x.shape[0]), x] 
alpha = 0.01 #Step size

iterations = 2000 #No. of iterations

m = y.size #No. of data points

np.random.seed(123) #Set the seed

theta = np.random.rand(2) #Pick some random values to start with

def gradient_descent(x, y, theta, iterations, alpha):

    past_costs = []

    past_thetas = [theta]

    for i in range(iterations):

        prediction = np.dot(x, theta)

        error = prediction - y

        cost = 1/(2*m) * np.dot(error.T, error)

        past_costs.append(cost)

        theta = theta - (alpha * (1/m) * np.dot(x.T, error))

        past_thetas.append(theta)        

    return past_thetas, past_costs

past_thetas, past_costs = gradient_descent(x, y, theta, iterations, alpha)

theta = past_thetas[-1]

print("Gradient Descent: {:.2f}, {:.2f}".format(theta[0], theta[1]))
plt.title('Cost Function J')

plt.xlabel('No. of iterations')

plt.ylabel('Cost')

plt.plot(past_costs)

plt.show()
fig = plt.figure()

ax = plt.axes()

plt.title('Sale Price vs Living Area')

plt.xlabel('Living Area in square feet (normalised)')

plt.ylabel('Sale Price ($)')

plt.scatter(x[:,1], y, color='red')

line, = ax.plot([], [], lw=2)

annotation = ax.text(-1, 700000, '')

annotation.set_animated(True)

plt.close()

fig