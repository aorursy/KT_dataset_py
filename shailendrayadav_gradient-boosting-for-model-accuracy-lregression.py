from sklearn.ensemble import GradientBoostingRegressor

from sklearn.linear_model import LinearRegression

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

df= pd.read_csv("../input/House_file.csv")

df.head()
reg=LinearRegression()

reg
reg.fit(df[["SqFt"]],df.Price)
y_pred=reg.predict([[2100]])
reg.score(df[["SqFt"]],df.Price)
gb=GradientBoostingRegressor()  
gb.fit(df[["SqFt"]],df.Price)
gb_predict=gb.predict([[2100]])
gb.score(df[["SqFt"]],df.Price)
mreg=LinearRegression()
mreg.fit(df[["SqFt","Bedrooms"]],df.Price)
mreg_predict=mreg.predict([[2000,2]])

mreg_predict
mreg.score(df[["SqFt","Bedrooms"]],df.Price)
mgb=GradientBoostingRegressor()

mgb
mgb.fit(df[["SqFt","Bedrooms"]],df.Price)
mgb_predict=mgb.predict([[2000,2]])

mgb_predict
mgb.score(df[["SqFt","Bedrooms"]],df.Price)