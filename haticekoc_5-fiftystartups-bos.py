import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
startups = pd.read_csv('../input/50_Startups.csv')

df = startups
df.head()
df.info()
df.shape
df.isna().sum()
df.corr()
corr = df.corr() 

sns.heatmap(corr, 

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values);
sns.scatterplot(x = "R&D Spend", y = "Profit", data = df);
sns.distplot(df["R&D Spend"], bins=20, color="red");

sns.distplot(df["Administration"], bins=20, color="green");

sns.distplot(df["Marketing Spend"], bins=20, color="blue");

sns.distplot(df["Profit"], bins=20, color="purple");
df.describe().T
df["State"].unique()
df['State'] = pd.Categorical(df['State'])
dfDummies = pd.get_dummies(df['State'], prefix = 'State')
dfDummies
df.drop(["State"], axis = 1, inplace = True)

dfDummies.drop(["State_New York"], axis = 1, inplace = True)
df = pd.concat([df, dfDummies], axis=1)
df
X = df.drop("Profit", axis = 1)

y = df["Profit"]
X
y
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.3)
X_train
X_test
y_train
y_test
from sklearn.linear_model import LinearRegression



lm= LinearRegression()
model = lm.fit(X_train, y_train)

model
y_pred = model.predict(X_train)
y_pred
model1 = lm.fit(X, y)

model1

predict_df =model1.predict(X)
df["Predict_Profit"] = model1.predict(X)

df
from sklearn.metrics import mean_absolute_error



MAE = mean_absolute_error(df["Profit"], df["Predict_Profit"])

MAE
from sklearn.metrics import mean_squared_error



MSE = mean_squared_error(df["Profit"], df["Predict_Profit"])

MSE
import math

RMSE = math.sqrt(MSE)

RMSE
model1.score(X,y)
import statsmodels.api as sm



stmodel = sm.OLS(y, X).fit()

stmodel.summary()