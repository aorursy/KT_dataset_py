import pandas as pd

import numpy as np


train = pd.read_csv("../input/train.csv")


test = pd.read_csv("../input/test.csv")
train.shape
train = train.dropna(axis = 0)
test.shape
test = test.dropna(axis=0)
x = train.iloc[:,:-1].values

y = train.iloc[:,1].values
x.dtype
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x,y)
test_x = test.iloc[:,0:1].values
y_pred= regressor.predict(test_x)
y_pred