import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import linear_model
import statsmodels.api as sm
import matplotlib.pyplot as plt
import os, random

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

df = pd.read_csv("../input/see19-a-comprehensive-global-covid19-dataset/see19-2020-06-16-11-26-31.csv")

df1 = df[['country','date','cases','wind-speed','humidity','temperature']].copy()

df1 = df1.dropna()

#AUS data
df2 = df1.loc[df1['country'] == "Australia"]

y_var = df2['cases']
x_var = df2[['temperature','wind-speed','humidity']]

lm = linear_model.LinearRegression()
model = lm.fit(x_var,y_var)

print('Intercept: \n', lm.intercept_)
print('Coefficients: \n', lm.coef_)

casescalc = lm.coef_[0]*1 + lm.coef_[1]*1 + lm.coef_[2]*1

print ("Every 1 degree rise in temperature, 1km increase in wind speed, 1% increase in humidity leads to a ",round(casescalc,0), " drop in cases")


