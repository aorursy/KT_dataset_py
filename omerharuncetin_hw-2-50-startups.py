import numpy as np

import seaborn as sns

import pandas as pd
startups = pd.read_csv("../input/50_Startups.csv")

df = startups.copy()
df.head()
df.info()
df.shape
df.isnull().sum()
df.corr()
sns.heatmap(df.corr())
sns.scatterplot(x = df["R&D Spend"], y = df["Profit"])
df.hist()
df.describe()
df["State"].unique()
df["State"] = pd.Categorical(df["State"])
stateDummies = pd.get_dummies(df["State"], prefix = "State")
df = pd.concat([df,stateDummies],axis = 1)
df.head()
df.drop(["State","State_Florida"],axis = 1, inplace = True)
df.head()
X = df.drop("Profit",axis = 1)

y = df["Profit"]
X
y
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 1/5)
X_train
X_test
y_train
y_test
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict([[52000,4789.14,20000,0,0]])

y_pred
df["pred_Profit"] = model.predict(X)

df.sample(10)
import sklearn.metrics as metrics

import math
MAE = metrics.mean_absolute_error(df["Profit"],df["pred_Profit"])

MAE
MSE = metrics.mean_squared_error(df["Profit"],df["pred_Profit"])

MSE
RMSE = math.sqrt(MSE)

RMSE
model.score(X_train,y_train)
sns.lmplot(x = "Profit", y = "pred_Profit", data = df); 
import statsmodels.api as sm
stmodel = sm.OLS(y,X).fit()
stmodel.summary()