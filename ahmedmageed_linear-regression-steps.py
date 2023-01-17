import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt 

%matplotlib inline
dataset_dirty = pd.read_csv("../input/train.csv")

dataset = dataset_dirty.dropna()

testset_dirty = pd.read_csv("../input/test.csv")

testset = testset_dirty.dropna()

testset
x_train = dataset.iloc[:,0].values.reshape(-1, 1)

y_train = dataset.iloc[:,1].values
from sklearn.linear_model import LinearRegression
model = LinearRegression()

model.fit(x_train, y_train)
x_test = testset.iloc[:,0].values.reshape(-1, 1)

y_test = testset.iloc[:,1].values
goal_test = model.predict(x_test)
plt.plot(x_test, goal_test, x_test, y_test, '.')

plt.title("Accuracy of the model")

plt.show()