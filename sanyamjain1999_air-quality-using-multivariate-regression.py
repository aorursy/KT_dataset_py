import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
x_test = pd.read_csv("../input/Test.csv")

Train = pd.read_csv("../input/Train.csv")
Train = Train.values

print(Train.shape)
X = Train[:,:5]

Y = Train[:,5]
print(X.shape)

print(Y.shape)
from sklearn.linear_model import LinearRegression
lr = LinearRegression(normalize = True)

lr.fit(X,Y)
xt = x_test.values

print(xt.shape)
y_pred = lr.predict(xt)
x_test = x_test.drop(['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5'], axis=1)

x_test['target'] = y_pred
x_test.to_csv('hardwork.csv', index=True)