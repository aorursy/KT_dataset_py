import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

df = pd.read_csv(filepath_or_buffer = "../input/advertising-dataset/advertising.csv")

print('First 5 data entries:\n',df.head())

print('Shape:',df.shape)

print('describe:\n',df.describe())
_=plt.scatter(df['TV'],df['Sales'],marker='*')

_=plt.xlabel('tv advertisment')

_=plt.ylabel('sales')

plt.show()
from sklearn import linear_model

import statsmodels.api as sm

x=df['TV']

y=df['Sales']
print("\n model  with statsmodel: \n")



x=sm.add_constant(x)

model=sm.OLS(y,x).fit()

print('Intercept:',model.params[0])

print('Coefficient: ',model.params[1])

print('r square:',model.rsquared)
print(" model with sklearn: \n") 



regression=linear_model.LinearRegression()

regression.fit(x,y)

print('Intercept:',regression.intercept_)

print('Coefficient:',regression.coef_)

print('r square:',regression.score(x,y))
print("Regression line:\n")

import seaborn as sns

_=sns.regplot(x=df['TV'],y=df['Sales'],data=df)

_=plt.xlabel('tv advertisment')

_=plt.ylabel('sales')

plt.show()

print("Residual plot:\n")

y_pred=regression.predict(x)

residual=y_pred-y

_=sns.distplot(residual)

plt.show()
residual_ss=residual**2

rmse=np.sqrt(residual_ss.mean())

rmse