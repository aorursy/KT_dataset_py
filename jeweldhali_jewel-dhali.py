import numpy as np

import pandas as pd

import math

import matplotlib.pyplot as plt

from sklearn import linear_model
data = "../input/homeprices1.csv"
df = pd.read_csv(data)
df
df.bedrooms.mean()
br = math.floor(df.bedrooms.mean())
br
df.bedrooms = df.bedrooms.fillna(br)
df
reg = linear_model.LinearRegression()

reg.fit(df[['area','bedrooms','age']], df.price)
reg.coef_
reg.intercept_
reg.predict([[3000,3,40]])
reg.predict([[3200,4,18]])