import pandas as pd

import numpy as np

import sqlite3
conn=sqlite3.connect('../input/database.sqlite')

data=pd.read_sql_query('SELECT * FROM Player_Attributes',conn)

data.head()
features = [

       'potential', 'crossing', 'finishing', 'heading_accuracy',

       'short_passing', 'volleys', 'dribbling', 'curve', 'free_kick_accuracy',

       'long_passing', 'ball_control', 'acceleration', 'sprint_speed',

       'agility', 'reactions', 'balance', 'shot_power', 'jumping', 'stamina',

       'strength', 'long_shots', 'aggression', 'interceptions', 'positioning',

       'vision', 'penalties', 'marking', 'standing_tackle', 'sliding_tackle',

       'gk_diving', 'gk_handling', 'gk_kicking', 'gk_positioning',

       'gk_reflexes']
data=data.dropna()
target ='overall_rating'
data[target].describe()
X=data[features].copy()
Y=data[target].copy()
X.sample()
Y.head()
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error
X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size=0.33, random_state=99)
reg = LinearRegression()

reg.fit(X_train,Y_train)
Y_pred = reg.predict(X_test)

Y_pred
Lin_Reg_RMSE = np.sqrt(mean_squared_error(y_true=Y_test, y_pred=Y_pred))
Lin_Reg_RMSE
from sklearn.tree import DecisionTreeRegressor
reg=DecisionTreeRegressor(max_depth=20)

reg.fit(X_train,Y_train)
Y_pred=reg.predict(X_test)

Y_pred
Dtree_Reg_RMSE = np.sqrt(mean_squared_error(y_true=Y_test, y_pred=Y_pred))
Dtree_Reg_RMSE
target_c='overall_level'

data[target_c]=3

data.loc[data[target]>=85,target_c]=1

data.loc[(data[target]<85) & (data[target]>=70),target_c]=2
data[[target,target_c]]
from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier
Y=data[target_c].copy()
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.33,random_state=99)
X_train.head()
clasif=DecisionTreeClassifier(max_leaf_nodes=15,random_state=0)

clasif.fit(X_train,Y_train)
pred=clasif.predict(X_test)
accuracy_score(y_true=Y_test,y_pred=pred)