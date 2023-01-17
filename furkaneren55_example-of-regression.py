import pandas as pd

import numpy as np

import matplotlib.pyplot as plt 

import seaborn as sns
startups= pd.read_csv("../input/50_Startups.csv",sep=",")

df=startups.copy()

df.head()
df.info()
df.shape
df.isnull().sum()
df.corr()
f,ax=plt.subplots(figsize=(10,10))

sns.heatmap(df.corr(),annot=True,linewidths=5,fmt='.0%',ax=ax)

plt.show()
plt.figure(figsize=(20,5))

sns.scatterplot( x='R&D Spend', y='Profit',color='red',data=df)

plt.title('R&D Spend- Profit')

plt.show()
plt.figure(figsize=(20,5))

sns.regplot( x='Marketing Spend', y='Profit',color='blue',data=df)

plt.title('R&D Spend- Profit')

plt.show()
df.hist(figsize = (10,10))

plt.show()
df.describe().T
df["State"].unique()
df['State'] = pd.Categorical(df['State'])

dfDummies = pd.get_dummies(df['State'], prefix = 'State')

dfDummies
df = pd.concat([df, dfDummies], axis=1)

df.head()
df=df.drop(["State"],axis=1)
df.head()
X = df.drop("Profit", axis = 1)

y = df["Profit"]
X
y
from sklearn.model_selection import train_test_split

x_train, x_test ,y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)
x_train
y_train
x_test
y_test
from sklearn.linear_model import LinearRegression



linear_regresyon = LinearRegression()
linear_regresyon.fit(X, y)
y_pred=linear_regresyon.predict(X)
df["y_pred"]=y_pred
df
df["Fark"]=df["Profit"]-df["y_pred"]
df
from sklearn.metrics import mean_squared_error



MSE = mean_squared_error(df["Profit"], df["y_pred"])

MSE
from sklearn.metrics import mean_absolute_error

MAE = mean_absolute_error(df["Profit"], df["y_pred"])

MAE
import math



RMSE = math.sqrt(MSE)

RMSE
linear_regresyon.score(X,y)
import statsmodels.api as sm
stmodel = sm.OLS(y, X).fit()
stmodel.summary()