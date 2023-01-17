import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

from sklearn.model_selection import train_test_split as tts

from sklearn.metrics import mean_squared_error

print(os.listdir("../input"))
df = pd.read_csv('../input/Housing Price data set.csv')

df.info()

df = df.replace({'yes' : 1, 'no':0})
df.columns
df = df.assign(b=1)

df.columns
x_features = ['b','lotsize', 'bedrooms', 'bathrms', 'stories']

x_df = df[x_features]

x_df.head()

y_df = df['price']

X_train, X_test, Y_train, Y_test = tts(x_df, y_df, test_size = 0.3, random_state = 5)
def normal_train(X, Y):

    transposeX = np.transpose(X)

    try:

        A = np.linalg.inv(np.dot(transposeX,X))

        B = np.dot(transposeX,Y)

        return np.dot(A,B)

    except numpy.linalg.LinAlgError:

        print("X is not invertible")

def normal_predict(X, weights):

    transposeW = np.transpose(weights)

    return np.dot(transposeW, X)
weights = normal_train(X_train, Y_train)

weights = weights.astype(float)

X_test = X_test.astype(float)
normal_predict(weights, X_test.iloc[0])
Y_predicted = [normal_predict(weights, x) for x in X_test.values]

mean_squared_error(Y_test, Y_predicted)