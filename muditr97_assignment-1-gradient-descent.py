import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

from sklearn.model_selection import train_test_split as tts

from sklearn.metrics import mean_squared_error

print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/Housing Price data set.csv')

df.info()

df = df.replace({'yes': 1, 'no': 0})
df.head()
df.columns
x_features = ['lotsize', 'bedrooms', 'bathrms', 'stories','driveway', 'recroom', 'fullbase', 'gashw', 'airco', 'garagepl','prefarea']
df.head()
x_df = df[x_features]
x_df.head()
y_df = df['price']

#y_df.head()

x_df.head()
X_train, X_test, Y_train, Y_test = tts(x_df, y_df, test_size = 0.4, random_state = 5)
m = len(Y_train)

alpha = 0.1

num_iterations = 1000

weights = np.zeros((12,1))
X_train = (X_train - np.mean(X_train))/np.std(X_train)

X_test = (X_test - np.mean(X_test))/np.std(X_test)

Y_train = Y_train[:,np.newaxis]

Y_test = Y_test[:,np.newaxis]
X_train = X_train.assign(b=1)

X_test = X_test.assign(b=1)

X_test.head()
Y_train
X_train.head()
Y_test
def GDM(x, y, weights, alpha, iterations):

    m = len(y)

    print(x.shape)

    print(y.shape)

    print(weights.shape)

    for _ in range(iterations):

        temp = np.dot(x, weights) - y

        temp = np.dot(x.T, temp)

        weights = weights - (alpha/m) * temp

    return weights
weights = GDM(X_train, Y_train, weights, alpha, num_iterations)
weights.shape
def GradientDescent_predict(x, weights):

    return np.dot(x, weights)
weights
GradientDescent_predict(X_train.iloc[0],weights)
Y_Predicted = [GradientDescent_predict(x,weights) for x in X_test.values]
mean_squared_error(Y_test, Y_Predicted)