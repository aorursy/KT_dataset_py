import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
x_test = pd.read_csv("../input/Linear_X_Test.csv")

X_Train = pd.read_csv("../input/Linear_X_Train.csv")

Y_train = pd.read_csv("../input/Linear_Y_Train.csv")
plt.scatter(X_Train,Y_train)

plt.show()
print(X_Train.shape)

print(Y_train.shape)
from sklearn.linear_model import LinearRegression
lr = LinearRegression()

lr.fit(X_Train,Y_train)
xt = x_test.values

y_pred = lr.predict(xt.reshape(-1,1))
x_test = x_test.drop(['x'], axis=1)
x_test['y'] = y_pred
x_test.to_csv('hardwork.csv', index=False)