import sqlite3
import pandas as pd

import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
cnx = sqlite3.connect('../input/soccer/database.sqlite')

df = pd.read_sql_query("SELECT * FROM Player_Attributes", cnx)
df.head()
df.shape
df.columns
features = [

       'potential', 'crossing', 'finishing', 'heading_accuracy',

       'short_passing', 'volleys', 'dribbling', 'curve', 'free_kick_accuracy',

       'long_passing', 'ball_control', 'acceleration', 'sprint_speed',

       'agility', 'reactions', 'balance', 'shot_power', 'jumping', 'stamina',

       'strength', 'long_shots', 'aggression', 'interceptions', 'positioning',

       'vision', 'penalties', 'marking', 'standing_tackle', 'sliding_tackle',

       'gk_diving', 'gk_handling', 'gk_kicking', 'gk_positioning',

       'gk_reflexes']
target = ['overall_rating']
df = df.dropna()
X = df[features]
y = df[target]
X.head()
X.iloc[2]
y
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33,random_state=324)
regressor = LinearRegression()

regressor.fit(X_train,y_train)
y_prediction = regressor.predict(X_test)
y_prediction
y_test.describe()
print(".............Evaluation metrics for Linear Regression..............")

from sklearn import metrics

score = regressor.score(X_test, y_test)

n=len(df[target])

p=len(features)

adjr= 1-(1-score)*(n-1)/(n-p-1)

print("RSquared: ",score)

print("AdjustedRSquared: ",adjr)

print('MAE', metrics.mean_absolute_error(y_test, y_prediction))

print('MSE', metrics.mean_squared_error(y_test, y_prediction))

print('RMSE', np.sqrt(metrics.mean_squared_error(y_test, y_prediction)))
regressor = DecisionTreeRegressor(max_depth=20)

regressor.fit(X_train,y_train)
y_prediction = regressor.predict(X_test)

y_prediction
y_test.describe()
print(".............Evaluation metrics for Decision tree Regression..............")

from sklearn import metrics

score = regressor.score(X_test, y_test)

n=len(df[target])

p=len(features)

adjr= 1-(1-score)*(n-1)/(n-p-1)

print("RSquared: ",score)

print("AdjustedRSquared: ",adjr)

print('MAE', metrics.mean_absolute_error(y_test, y_prediction))

print('MSE', metrics.mean_squared_error(y_test, y_prediction))

print('RMSE', np.sqrt(metrics.mean_squared_error(y_test, y_prediction)))