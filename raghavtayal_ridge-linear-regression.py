import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split as tts

from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt

import seaborn as sns

import os

print(os.listdir("../input"))
df = pd.read_csv('../input/Housing Price data set.csv')
df.info()
df = df.replace({'yes': 1, 'no': 0})

df.head()
df.columns

x_features = ['lotsize', 'bedrooms', 'bathrms', 'stories','driveway', 'recroom', 'fullbase', 'gashw', 'airco', 'garagepl',

       'prefarea']
x_df = df[x_features]

x_df.head()

y_df = df['price']

X_train, X_test, Y_train, Y_test = tts(x_df, y_df, test_size = 0.3, random_state = 5)
m = len(Y_train)

alpha = 0.01

num_iters = 800

weights = np.zeros((12,1))

X_train = (X_train - np.mean(X_train))/np.std(X_train)

X_test = (X_test - np.mean(X_test))/np.std(X_test)

Y_train = Y_train[:,np.newaxis]

Y_test = Y_test[:,np.newaxis]
X_train = X_train.assign(b=1)

X_test = X_test.assign(b=1)

# X_train.head()

X_test.head()

# Y_train

# Y_test
def gradientDescentMulti(X, y, weights, alpha, iterations, lamda):

    m = len(y)

    print(X.shape)

    print(y.shape)

    print(weights.shape)

    for _ in range(iterations):

        temp = np.dot(X, weights) - y

        temp = np.dot(X.T, temp)

        weights = (weights*(1-alpha*(lamda/m))) - ((alpha/m) * temp)

    return weights
weights = gradientDescentMulti(X_train, Y_train, weights, alpha, num_iters,100)
weights.shape
def GD_predict(X, weights):

    return np.dot(X, weights)
weights
GD_predict(X_test.iloc[0], weights)
Y_predicted = [GD_predict(x, weights) for x in X_test.values]

mean_squared_error(Y_test, Y_predicted)
Y_test[0]