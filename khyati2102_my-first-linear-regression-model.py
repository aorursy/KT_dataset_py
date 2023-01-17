#importing all the lib's

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
#importing the data

test = pd.read_csv("../input/random-linear-regression/test.csv")

train = pd.read_csv("../input/random-linear-regression/train.csv")
#clean the data

train=train.dropna()

#shape of the data

print("the shape is {}".format(train.shape))
x_train=train.as_matrix(['x'])

y_train=train.as_matrix(['y'])

x_test=test.as_matrix(['x'])

y_test=test.as_matrix(['y'])
from sklearn.linear_model import LinearRegression

lr=LinearRegression()

#fit the data

lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)

#print("accuracy in training set {}".format(lr.score(x_train,y_train)))

print("accuracy : {}".format(lr.score(x_test,y_test)))

plt.plot(y_pred,'.',label="y_pred")

plt.plot(y_test,'*',label="y_test")

plt.xlabel("Predicted Y")

plt.ylabel("Actual Y")

plt.title("Relationship between Actual Target and Predicted Target")

plt.legend(loc="upper right")

plt.show()