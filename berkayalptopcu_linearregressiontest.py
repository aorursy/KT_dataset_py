import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
startups= pd.read_csv("../input/50-startups/50_Startups.csv")

df=startups.copy()
df
df.head()
df.info()
df.dtypes
df.shape
df.isnull().sum()
corr = df.corr()

corr
sns.heatmap(corr, 

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values);
sns.set(rc={'figure.figsize':(8,8)})
sns.scatterplot(df["R&D Spend"], df["Profit"]);
df.hist(figsize = (8,8))

plt.show()
df.describe().T
df.State.unique()
df["State"] = pd.Categorical(df["State"])
dfDummies = pd.get_dummies(df["State"],prefix="state")

dfDummies.head()
df=pd.concat([df,dfDummies],axis=1)
df.head()
df=df.drop(columns="state_Florida")
df.head()
df=df.drop(columns="State")
df.head()
y= df['Profit']
X=df.drop(['Profit'], axis=1)
X.head()
y.head()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 35)
X_train.info()
X_test.info()
y_train
y_test
from sklearn.linear_model import LinearRegression

linear_regresyon = LinearRegression()

linear_regresyon.fit(X,y)
y_pred=linear_regresyon.predict(X)
df["y_pred"]=y_pred
df.head()
df["pFark"]=df["Profit"]-df["y_pred"]

df
from sklearn.metrics import mean_absolute_error



MSA = mean_absolute_error(df["Profit"], df["y_pred"])

MSA
from sklearn.metrics import mean_squared_error



MSE = mean_squared_error(df["Profit"], df["y_pred"])

MSE
import math



RMSE = math.sqrt(MSE)

RMSE
linear_regresyon.score(X,y)