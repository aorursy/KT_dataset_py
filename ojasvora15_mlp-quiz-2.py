import pandas as pd

import numpy as np

import scipy

import matplotlib.pyplot as plt

from sklearn import linear_model as ln

import os

print(os.listdir("../input"))
df1 = pd.read_csv("../input/cats.csv")

df1.head(10)
x = df1[["Bwt"]]

y = df1[["Hwt"]]

model = ln.LinearRegression()

results = model.fit(x,y)



print(model.intercept_, model.coef_)
df2 = pd.read_csv("../input/boston.csv")

df2.head(10)
x = df2[["age"]]

y = df2[["medv"]]

model = ln.LinearRegression()

results = model.fit(x,y)



print(model.intercept_, model.coef_)