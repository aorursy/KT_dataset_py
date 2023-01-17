import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import statsmodels.api as sm

import seaborn as sns

import math
df = pd.read_csv('../input/students-data-for-mlr/amsPredictionSheet1-201009-150447.csv')

df.head()
df.describe()
corr = df.corr()

corr.style.background_gradient()
endog = df['ESE']

exog = sm.add_constant(df[['MSE','Attendance','HRS']])

print(exog)
X = exog.to_numpy()

Y = endog.to_numpy()

xt = np.transpose(X)

print(xt)
multi = np.matmul(xt,X)

print(multi)
inv = np.linalg.inv(multi)

multi2 = np.matmul(inv,xt)

print(multi2)
beta = np.matmul(multi2,Y)

print(beta)
model = sm.OLS(endog,exog)

results = model.fit()

print(results.summary())
from sklearn import linear_model

x = df[['MSE','Attendance','HRS']]

y = df['ESE']



lm = linear_model.LinearRegression()

mod = lm.fit(x,y)

lm.coef_
lm.intercept_