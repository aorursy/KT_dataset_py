import sqlite3
import pandas as pd
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
RMSE = sqrt(mean_squared_error(y_true = y_test, y_pred = y_prediction))
print(RMSE)

RMSE
regressor = DecisionTreeRegressor(max_depth=20)

regressor.fit(X_train,y_train)
y_prediction = regressor.predict(X_test)

y_prediction
y_test.describe()
RMSE = sqrt(mean_squared_error(y_true = y_test, y_pred = y_prediction))
print(RMSE)