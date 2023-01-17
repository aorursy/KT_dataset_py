import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/df.csv")
df.head()
df.info()
df.fillna(0,inplace = True)
df['RainTomorrow'].value_counts()
X = df.drop("RainTomorrow",axis = 1)

y = df['RainTomorrow']
X_arr = X.as_matrix()
from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y = train_test_split(X_arr,y,random_state = 34,test_size = 0.25)
train_y.shape
'''tr_y = train_y.values.reshape((11341,1))'''
class LogisticRegression_new:

    def __init__(self, lr=0.01, n_iter=1000):

        self.lr = lr

        self.n_iter = n_iter



    def predict(self, X):

        X = self.normalize(X)

        linear = self._linear(X)

        preds = self._non_linear(linear)

        return (preds >= 0.5).astype('int')



    def _non_linear(self, X):

        return 1 / (1 + np.exp(-X))



    def _linear(self, X):

        return np.dot(X, self.weights) + self.bias



    def initialize_weights(self, X):

        # We have same number of weights as number of features

        self.weights = np.random.rand(X.shape[1], 1)

        # we will also add a bias to the terms that

        # can be interpretted as y intercept of our model!

        self.bias = np.zeros((1,))



    def fit(self, X_train, Y_train):

        self.initialize_weights(X_train)



        # get mean and stddev for normalization

        self.x_mean = X_train.mean(axis=0).T

        self.x_stddev = X_train.std(axis=0).T



        # normalize data

        X_train = self.normalize(X_train)



        # Run gradient descent for n iterations

        for i in range(self.n_iter):

            # make normalized predictions

            probs = self._non_linear(self._linear(X_train))

            diff = probs - Y_train.values.reshape((len(Y_train),1))



            # d/dw and d/db of mse

            delta_w = np.mean(diff * X_train, axis=0, keepdims=True).T

            delta_b = np.mean(diff)



            # update weights

            self.weights = self.weights - self.lr * delta_w

            self.bias = self.bias - self.lr * delta_b

        return self



    def normalize(self, X):

        X = (X - self.x_mean) / self.x_stddev

        return X



    def accuracy(self, X, y):

        preds = self.predict(X)

        return np.mean(preds == y)



    def loss(self, X, y):

        probs = self._non_linear(self._linear(X))



        # entropy when true class is positive

        pos_log = y * np.log(probs + 1e-15)

        # entropy when true class is negative

        neg_log = (1 - y) * np.log((1 - probs) + 1e-15)



        l = -np.mean(pos_log + neg_log)

        return l
lr = LogisticRegression_new()

lr.fit(train_x, train_y)
pred = lr.predict(test_x)
from sklearn.metrics import accuracy_score
accuracy_score(test_y,pred)
### Using default logistic regression in sklearn

from sklearn.linear_model import LogisticRegression

log = LogisticRegression()

log.fit(train_x,train_y)

predi = log.predict(test_x)

accuracy_score(test_y,predi)