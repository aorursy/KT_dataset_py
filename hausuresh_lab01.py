# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import matplotlib.pyplot as plt

import sklearn.linear_model

import sklearn.preprocessing

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Make data

X_train = np.random.rand(1000,1)

Y_train = 24 + 5*X_train + np.random.randn(1000,1)*0.9

#Y_train = 5 + 2*X_train + 3*(X_train**2) + 4*(X_train**3) + 2*(X_train**4) - X_train**5

X_test = np.random.rand(100,1)

Y_test = 24 + 5*X_test + np.random.randn(100,1)*0.9

#Y_test = 5 + 2*X_test + 3*(X_test**2) + 4*(X_test**3) + 2*(X_test**4) - X_test**5 
X_train.shape
Y_train.shape
#Plot Training data

plt.scatter(X_train,Y_train);

# Add naming to each axis

plt.xlabel("$X$", fontsize=18);

plt.ylabel("$y$", rotation=0, fontsize=18);
#Plot Testing data over Training

plt.plot(X_train,Y_train,"o",alpha = 0.4,label = "train")

plt.plot(X_test,Y_test,"o",alpha = 0.7,label = "test")

plt.title("Data Distribution")

plt.legend()

plt.show()
from sklearn import linear_model

model = linear_model.LinearRegression()

model.fit(X_train,Y_train)
print(f"Coef is {model.coef_}, Intercept is {model.intercept_}")
# Visualize the trained results

plt.plot(X_train,Y_train,"o",alpha = 0.4,label = "train")

plt.plot(X_test,Y_test,"o",label = "test")

plt.plot(X_train, X_train*model.coef_ + model.intercept_,label = "fitting line trained")

plt.legend()

plt.show()
Y_predict = model.predict(X_test)

# sklearn LinearRegressor() provide score() function to estimate loss of learned regression model

print(f"Score of our model: {model.score(X_test,Y_test)}")
## Calculate loss 

from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_absolute_error

#L2 MSE loss

mse_error = mean_squared_error(Y_test,Y_predict)

#L1 Absolute loss

l1_error = mean_absolute_error(Y_test, Y_predict)

print(f"MSE error is {rmse_error}, MAE (absolute) error is {l1_error}")
# visualize the trained results

plt.plot(X_test,Y_predict,"o",label = "predict")

plt.plot(X_test,X_test * model.coef_ + model.intercept_,label = "predict with coef, intercept")

plt.plot(X_test,Y_test,"o",alpha = 0.5,label = "test data")

plt.legend()

plt.show()
from sklearn import linear_model

model_logistic = linear_model.LogisticRegression()
cov = [[1, 0], [0, 1]]

mean0 = [2, 2]

mean1 = [5, 7]

N = 1000



# generate train set

# class 0

# generate class 0 samples x axis and y axis values from gaussian distribution

# numpy provides multivariate_normal() function to generate multiple values at once

X0_train = np.random.multivariate_normal(mean0, cov, N)

# class 0 have labels 0

Y0_train = np.zeros(N)

# class 1

# generate class 1 samples x axis and y axis values from gaussian distribution

X1_train = np.random.multivariate_normal(mean1, cov, N)

# class 1 have labels 1

Y1_train = np.ones(N)

# concatenate the training data for each class to create training set with both class.

X_train = np.concatenate([X0_train, X1_train], axis=0)

Y_train = np.concatenate([Y0_train, Y1_train], axis=0)
#Visualize

plt.plot(X0_train[:,0],X0_train[:,1],"o",alpha = 0.5,label = "Class 0")

plt.plot(X1_train[:,0],X1_train[:,1],"o",alpha = 0.5,label = "Class 1")

plt.legend()

plt.show()
model_logistic.fit(X_train,Y_train)

print(f"Coef: {model_logistic.coef_}, interception: {model_logistic.intercept_}")

model_logistic.score(X_train,Y_train)
#Predict

Y_predict = model_logistic.predict(X_train)
#Visualize result after trained

plt.plot(X0_train[:,0], X0_train[:,1],"o", alpha = 0.5, label = "Class 0 train")

plt.plot(X1_train[:,0], X1_train[:,1],"o", alpha = 0.5, label = "Class 1 train")

plt.plot(X_train[:,0], (0-model_logistic.intercept_ - model_logistic.coef_[0, 0]*X_train[:, 0])/model_logistic.coef_[0, 1], label="classification boundary")

plt.legend() 

plt.show()
model_logistic.score(X_train, Y_train)
#from sklearn import multiclass

model_perceptron =linear_model.Perceptron()
model_perceptron.fit(X_train, Y_train)
model_perceptron.score(X_train,Y_train)