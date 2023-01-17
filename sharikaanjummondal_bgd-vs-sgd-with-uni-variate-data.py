# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#our hypothesis function 



def hypothesis(theta, X):

    h = np.ones((X.shape[0],1))

    for i in range(0,X.shape[0]):

        x = np.concatenate((np.ones(1), np.array([X[i]])), axis = 0)

        h[i] = float(np.matmul(theta, x))

    h = h.reshape(X.shape[0])

    return h
#our SGD function



def SGD(theta, alpha, num_iters, h, X, y):

    for i in range(0,num_iters):

        theta[0] = theta[0] - (alpha) * (h - y)

        theta[1] = theta[1] - (alpha) * ((h - y) * X)

        h = theta[1]*X + theta[0] 

    return theta
#SGD function integrated with Linear Regression



def sgd_linear_regression(X, y, alpha, num_iters):

    # initializing the parameter vector...

    theta = np.zeros(2)

    # hypothesis calculation....

    h = hypothesis(theta, X)    

    # returning the optimized parameters by Gradient Descent...

    for i in range(0, X.shape[0]):

        theta = SGD(theta,alpha,num_iters,h[i],X[i],y[i])

    theta = theta.reshape(1, 2)

    return theta
#our BGD function



def BGD(theta, alpha, num_iters, h, X, y):

    cost = np.ones(num_iters)

    theta_0 = np.ones(num_iters)

    theta_1 = np.ones(num_iters)

    for i in range(0,num_iters):

        theta[0] = theta[0] - (alpha/X.shape[0]) * sum(h - y)

        theta[1] = theta[1] - (alpha/X.shape[0]) * sum((h - y) * X)

        h = hypothesis(theta, X)

        cost[i] = (1/X.shape[0]) * 0.5 * sum(np.square(h - y))

        theta_0[i] = theta[0]

        theta_1[i] = theta[1]

    theta = theta.reshape(1,2)

    return theta, theta_0, theta_1, cost
#BGD function integrated with linear Regression



def linear_regression(X, y, alpha, num_iters):

    # initializing the parameter vector...

    theta = np.zeros(2)

    # hypothesis calculation....

    h = hypothesis(theta, X)    

    # returning the optimized parameters by Gradient Descent...

    theta,theta_0,theta_1,cost= BGD(theta,alpha,num_iters,h,X,y)

    return theta, theta_0, theta_1, cost
# loading our uni-variate dataset



data = pd.read_csv('/kaggle/input/dataset/data.csv',header=None)

data.head()
# extracting features and labels



X = data.iloc[:,0].values #the feature_set

y = data.iloc[:,1].values#the labels
#visualising my features and labels



import matplotlib.pyplot as plt

plt.scatter(X,y)

plt.xlabel('Population of City in 10,000s')

plt.ylabel('Profit in $10,000s')
import matplotlib.pyplot as plt 

# getting the predictions...

theta = sgd_linear_regression(X, y, 0.0001, 10000)

training_predictions = hypothesis(theta, X)

scatter = plt.scatter(X, y, label="training data")

regression_line = plt.plot(X, training_predictions

                           , label="linear regression")

plt.legend()

plt.xlabel('X axis')

plt.ylabel('y axis')

plt.title('Regression line with SGD')
from sklearn.metrics import r2_score   #for performance analysis

print('R2 score for SGD',r2_score(y,training_predictions))
theta,theta_0,theta_1,cost=linear_regression(X,y,0.0001,250)



#predictions

training_predictions = hypothesis(theta, X)

scatter = plt.scatter(X, y, label="training data")

regression_line = plt.plot(X, training_predictions, label="linear regression")

plt.legend()

plt.xlabel('X axis')

plt.ylabel('y axis')

plt.title('Regression line with BGD')
from sklearn.metrics import r2_score   #for performance analysis

print('R2 score for BGD',r2_score(y,training_predictions))