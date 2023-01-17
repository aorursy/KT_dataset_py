# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import datasets

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn import preprocessing

from sklearn.linear_model import LinearRegression
# load booston dataset from sklearn dataset

boston = datasets.load_boston()

boston.keys()
df = pd.DataFrame(data = boston.data)

df.columns = boston.feature_names

df['target'] = boston.target
df.head()
# checking correlation of all variable with target variable

corr = df.corr()

corr['target'].sort_values(ascending =False)
# preprocessing df dataset for gradient descent impelementaion

y = preprocessing.scale(df['target'])

x = df.drop('target', 1)

# Standardize all column

x = preprocessing.scale(x)

x = np.c_[np.ones(x.shape[0]), x]

x.shape
x
# this function get x, y, epoch, alpha and theta values and return updated cost function,

# updated theta, cost_list and preiction list

def gradient_descent(x, y, epoch, alpha, theta):

    cost_list = []

    prediction_list = []

    n = float(len(y))

    for i in range(epoch):

        prediction = np.dot(x, theta)

        prediction_list.append(prediction)

        error = prediction - y

        cost = 1/(2*n) * np.dot(error.T, error)

        cost_list.append(cost)

        theta = theta - (alpha * (1/n) * np.dot(x.T, error))

    return cost, theta, cost_list, prediction_list
np.random.seed(123) #Set the seed

theta = np.random.rand(14) # random assignment of theta for begin 



epoch = 10000

alpha = 0.01

cost, theta, cost_list, pred_list= gradient_descent(x, y, epoch, alpha, theta)
theta
# ploting cost values for each epoch

plt.title('Cost Function', size = 30)

plt.xlabel('No. of iterations', size=20)

plt.ylabel('Cost', size=20)

plt.plot(cost_list)

plt.show()
# Predict y values using gradient descent coefficients

y_pred = np.dot(x, theta)
# Mean square of residuals for gradient descent

mse_gd = ((pred_list[-1]-y)**2).mean()  #From Gradient Descent

print(f'Mean Square Error from Gradient Descent prediction : {round(mse_gd, 3)}')
lr = LinearRegression()

#Fitting the model

lr = lr.fit(x,y)

# prediction

lr_pred = lr.predict(x)
# coefficient values derived from sklearn linear regression 

lr.coef_
# R squared value using sklern model

r2 = lr.score(x,y)

print(f'R square from scikit learn: {round(r2, 3)}')
# R squared value using gradient descent

r2 = 1 - (sum((y - pred_list[-1])**2)) / (sum((y - y.mean())**2))

print(f'R square doing from the scratch: {round(r2, 3)}')