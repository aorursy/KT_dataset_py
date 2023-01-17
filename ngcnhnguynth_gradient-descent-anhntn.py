# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
!pip install numpy
import numpy as np

import matplotlib.pyplot as plt
#Y = 2X - 5

X = np.random.rand(400,1)

Y = -5 +2 * X+np.random.randn(400,1)*1.2
plt.plot(X,Y,'g.')

plt.xlabel("$x$", fontsize=15, fontweight='bold')

plt.ylabel("$y$", rotation=0, fontsize=15, fontweight='bold')
def cal_cost(theta,X,Y):

    m = len(Y)

    predictions = X.dot(theta)

    cost = 1/(2*m) * np.sum(np.square(predictions-Y))

    return cost
def gradient_descent(X,Y,theta,learning_rate=0.01,iterations=1500):

    m = len(Y)

    cost_history = np.zeros(iterations)

    theta_history = np.zeros((iterations,2))

    for it in range(iterations):

        

        prediction = np.dot(X,theta)

        

        theta = theta -(1/m)*learning_rate*( X.T.dot((prediction - Y)))

        theta_history[it,:] =theta.T

        cost_history[it]  = cal_cost(theta,X,Y)

        

    return theta, cost_history, theta_history
lr =0.01

n_iter = 1500



theta = np.random.randn(2,1)



X_b = np.c_[np.ones((len(X),1)),X]

theta,cost_history,theta_history = gradient_descent(X_b,Y,theta,lr,n_iter)





print('Theta0:          {:0.3f},\nTheta1:          {:0.3f}'.format(theta[0][0],theta[1][0]))

print('Final cost/MSE:  {:0.3f}'.format(cost_history[-1]))
fig,ax = plt.subplots(figsize=(12,8))



ax.set_ylabel('J(Theta)')

ax.set_xlabel('Iterations')

_=ax.plot(range(n_iter),cost_history,'b.')
fig,ax = plt.subplots(figsize=(10,8))

_=ax.plot(range(200),cost_history[:200],'b.')