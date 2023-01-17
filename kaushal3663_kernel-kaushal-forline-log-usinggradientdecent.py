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


import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (12.0, 9.0)



# Preprocessing Input data

Linear = pd.read_csv('/kaggle/input/mou-vally/MyData.csv')
Linear.head()
X = Linear.iloc[:, 0]

Y = Linear.iloc[:, 1]

plt.scatter(X, Y)

plt.show()
# Building the model

m = 0

c = 0



L = 0.0001  # The learning Rate

epochs = 1000  # The number of iterations to perform gradient descent



n = float(len(X)) # Number of elements in X



# Performing Gradient Descent 

for i in range(epochs): 

    Y_pred = m*X + c  # The current predicted value of Y

    D_m = (-2/n) * sum(X * (Y - Y_pred))  # Derivative wrt m

    D_c = (-2/n) * sum(Y - Y_pred)  # Derivative wrt c

    m = m - L * D_m  # Update m

    c = c - L * D_c  # Update c

    

print (m, c)
# Making predictions

Y_pred = m*X + c



plt.scatter(X, Y)

plt.plot([min(X), max(X)], [min(Y_pred), max(Y_pred)], color='red') # predicted

plt.show()
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from math import exp

plt.rcParams["figure.figsize"] = (10, 6)

Logistic = pd.read_csv("/kaggle/input/mysecdata/network.csv")

Logistic.head()
plt.scatter(Logistic['Age'], Logistic['Purchased'])

plt.show()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(Logistic['Age'], Logistic['Purchased'], test_size=0.20)
 #Helper function to normalize data

def normalize(X):

    return X - X.mean()
# Method to make predictions

def predict(X, b0, b1):

    return np.array([1 / (1 + exp(-1*b0 + -1*b1*x)) for x in X])
# Method to train the model

def logistic_regression(X, Y):



    X = normalize(X)



    # Initializing variables

    b0 = 0

    b1 = 0

    L = 0.001

    epochs = 300



    for epoch in range(epochs):

        y_pred = predict(X, b0, b1)

        D_b0 = -2 * sum((Y - y_pred) * y_pred * (1 - y_pred))  # Derivative of loss wrt b0

        D_b1 = -2 * sum(X * (Y - y_pred) * y_pred * (1 - y_pred))  # Derivative of loss wrt b1

        b0 = b0 - L * D_b0

        b1 = b1 - L * D_b1

    

    return b0, b1
# Training the model

b0, b1 = logistic_regression(X_train, y_train)



# Making predictions

# X_test = X_test.sort_values()  # Sorting values is optional only to see the line graph

X_test_norm = normalize(X_test)

y_pred = predict(X_test_norm, b0, b1)

y_pred = [1 if p >= 0.5 else 0 for p in y_pred]



plt.clf()

plt.scatter(X_test, y_test)

plt.scatter(X_test, y_pred, c="red")

# plt.plot(X_test, y_pred, c="red", linestyle='-', marker='o') # Only if values are sorted

plt.show()
# The accuracy

accuracy = 0

for i in range(len(y_pred)):

    if y_pred[i] == y_test.iloc[i]:

        accuracy += 1

print(f"Accuracy = {accuracy / len(y_pred)}")