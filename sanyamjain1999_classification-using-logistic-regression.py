import numpy as np

import pandas as pd

from sklearn.linear_model import LogisticRegression
x_test = pd.read_csv("../input/Logistic_X_Test.csv")

X_Train = pd.read_csv("../input/Logistic_X_Train.csv")

Y_train = pd.read_csv("../input/Logistic_Y_Train.csv")
X = X_Train.values

Y = Y_train.values

xt = x_test.values
print(X.shape)

print(Y.shape)

Y = Y.reshape(3000,)

print(Y.shape)

print(xt.shape)
model = LogisticRegression(solver='sag')
model.fit(X, Y)
y_pred = model.predict(xt)
x_test = x_test.drop(['f1', 'f2', 'f3'], axis = 1)

x_test['label'] = y_pred
x_test.to_csv('chemicals.csv', index=False)