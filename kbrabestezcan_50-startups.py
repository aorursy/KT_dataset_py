import numpy as np

import seaborn as sns

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_absolute_error

import math

import statsmodels.api as sm
df = pd.read_csv("../input/startupscsv/50_Startups.csv", sep = ",")
df.head()
df.info()
df.shape
df.isnull().sum()
df.corr() # En güçlü ilişkinin R&D Spend ile Profit arasında olduğu görülüyor
sns.heatmap(df.corr(), annot = True)
sns.scatterplot(df["R&D Spend"], df["Profit"]);
df.hist()
df.describe().T
df["State"].unique()
df['State'] = pd.Categorical(df['State'])
dfDummies = pd.get_dummies(df['State'])
dfDummies.head()
df = pd.concat([df, dfDummies], axis=1)

df = df.drop(['State', "New York"], axis = 1)

df.head()
X = df.drop("Profit", axis = 1)

y = df["Profit"]
X.head()
y.head()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 3512, shuffle=1)
X_train
X_test
y_train
y_test


lm = LinearRegression()
model = lm.fit(X_train, y_train)
y_pred = lm.predict(X_test)
df_comp = pd.DataFrame({'Gerçek': y_test, 'Tahmin': y_pred})

df_comp


MSE = mean_squared_error(y_test, y_pred)

MSE
RMSE = math.sqrt(MSE)

RMSE


MAE = mean_absolute_error(y_test, y_pred)

MAE
model.score(X, y)


stmodel = sm.OLS(y, X).fit()

stmodel.summary()