# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk(b'/kaggle/input/Simple Linear Regression/'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Loading dataset

def load_data():

    dataset = pd.read_csv('../input/simple-linear-regression/Salary_Data.csv')

    return dataset
import matplotlib.pyplot as plt

#splitting the data into independent varibales (X) and Target variable (Y)

def dataset_XY():

    dataset = load_data()

    X = dataset.iloc[:, :-1].values

    Y = dataset.iloc[:, 1].values

    return [X,Y]
dataset = load_data()

X, Y = dataset_XY()

dataset.head()

#Finding empty values in dataset

print(dataset.isnull().sum())
from sklearn.model_selection import train_test_split

def splitXYTrain(X, Y, testSize):

    #Spkitting the data set into train and test data

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=testSize, random_state = 0)

    return [x_train, x_test, y_train, y_test]
x_train, x_test, y_train, y_test = splitXYTrain(X, Y, 0.2)
#Calculating mean of x_train and mean of y_train

def xy_mean(x_train, y_train):

    x_mean = sum(x_train)/len(x_train)

    y_mean = sum(y_train)/len(y_train)

    return [x_mean, y_mean]
x_mean, y_mean = xy_mean(x_train, y_train)

print(x_mean)

print(y_mean)
#finding coefficients (theta0, theta1) or (b0, b1)

numer = 0

denom = 0

m = len(x_train)

for i in range(m):

    numer += (x_train[i] - x_mean) * (y_train[i] - y_mean)

    denom += (x_train[i] - x_mean)**2

b1 = numer / denom

b0 = y_mean - (b1*x_mean)

print(b1, b0)
#finding predictions of training data

y_pred = b0 + b1*x_train

print(y_pred)
#plotting simple linear regression

plt.scatter(x_train, y_train)

plt.plot(x_train, y_pred, color='r')

plt.show()
#R squared

numer = 0.0

denom = 0.0

y_mean1 = 0

for i in range(len(y_train)):

    y_mean1 = y_mean1 + y_train[i]

y_mean1 = y_mean1/len(y_train)

for i in range(len(y_train)):

    numer = numer + ((y_pred[i] - y_mean1)**2 )

    denom = denom + ((y_train[i] - y_mean1)**2)

Rsquared = numer / denom

print(Rsquared)
#finding predictions of test data

y_pred_test = b0 + b1*x_test

print(y_pred_test)
#plotting simple linear regression

plt.scatter(x_test, y_test)

plt.plot(x_test, y_pred_test, color='r')

plt.show()
#R squared

numer = 0.0

denom = 0.0

y_mean1 = 0

for i in range(len(y_test)):

    y_mean1 = y_mean1 + y_test[i]

y_mean1 = y_mean1/len(y_test)

for i in range(len(y_test)):

    numer = numer + ((y_pred_test[i] - y_mean1)**2 )

    denom = denom + ((y_test[i] - y_mean1)**2)

Rsquared = numer / denom

print(Rsquared)