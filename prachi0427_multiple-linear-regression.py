import numpy as np

import pandas as pd

df=pd.read_csv('../input/real-estate-price-prediction/Real estate.csv')

df=df.drop('No',axis=1)

print("First 5 rows of data:\n",df.head())

print("Shape:",df.shape)

print("Describe:\n",df.describe())

print("missing values:\n",df.isnull().sum())
import matplotlib.pyplot as plt

import seaborn as sns

X=df[['X1 transaction date','X2 house age','X3 distance to the nearest MRT station',

     'X4 number of convenience stores','X5 latitude','X6 longitude']]

Y=df['Y house price of unit area']

g=sns.PairGrid(df)

g.map(plt.scatter)

plt.show()

import statsmodels.api as sm

X=sm.add_constant(X)

model= sm.OLS(Y,X).fit()

model.summary()
X1=df[['X1 transaction date','X2 house age','X3 distance to the nearest MRT station',

     'X4 number of convenience stores','X5 latitude']]

X1=sm.add_constant(X1)

model1= sm.OLS(Y,X1).fit()

model1.summary()

print("Residual plot:\n")

Y_pred=model1.predict(X1)

residual=Y_pred-Y

_=sns.distplot(residual)

plt.show()
residual_ss=residual**2

rmse=np.sqrt(residual_ss.mean())

rmse