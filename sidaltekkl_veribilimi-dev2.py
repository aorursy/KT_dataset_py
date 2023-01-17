import numpy as np
import seaborn as sns
import pandas as pd

df = pd.read_csv("../input/sp-startup/50_Startups.csv", sep = ",")

df.head()
df.info()
df.shape
df.isnull().sum()

df.corr()
sns.heatmap(df.corr(), annot = True)
sns.scatterplot(df["R&D Spend"], df["Profit"]);
df.hist (figsize =(10,12))

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
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 3512, shuffle=1)
X_train.head()
X_test.head()
y_train.head()
y_test.head()
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
model = lm.fit(X_train, y_train)
y_pred = lm.predict(X_test)
df_comp = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df_comp
from sklearn.metrics import mean_squared_error

MSE = mean_squared_error(y_test, y_pred)
MSE
from sklearn.metrics import mean_absolute_error

MSA = mean_absolute_error(y_test, y_pred)
MSA
import math

RMSE = math.sqrt(MSE)
RMSE
model.score(X, y)
import statsmodels.api as sm
stmodel = sm.OLS(y, X).fit()
stmodel.summary()