import pandas as pd

import numpy as np

import seaborn as sns 

import math

import matplotlib.pyplot as plt

import statsmodels.api as sm
df = pd.read_csv("../input/predictingese/amsPrediction - Sheet1.csv")

df.head()
df.describe()
corr=df.corr()

corr.style.background_gradient(cmap='coolwarm')
endog = df['ESE']

exog = sm.add_constant(df[['MSE','Attendance','HRS']])

print(exog)
X=exog.to_numpy()

Y= endog.to_numpy()

s1_xt =np.transpose(X)

print(s1_xt)
s2_mul1= np.matmul(s1_xt,X)

print(s2_mul1)
s3_inv=np.linalg.inv(s2_mul1)

print(s3_inv)
s4_mul= np.matmul(s3_inv,s1_xt)

print(s4_mul)
s5_res =np.matmul(s4_mul,Y)

print(s5_res)
mod = sm.OLS(endog, exog)

results = mod.fit()

print (results.summary())
from sklearn import linear_model

X = df[['MSE','Attendance','HRS']]

y = df['ESE']



lm = linear_model.LinearRegression()

model = lm.fit(X,y)

lm.coef_
lm.intercept_