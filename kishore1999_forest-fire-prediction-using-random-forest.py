import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestRegressor
data = pd.read_csv("../input/forest-fire-area/forestfires.csv")

data.head()
#missing values

data.isna().sum().sum()
#label encoding

le = LabelEncoder()

data['month'] = le.fit_transform(data['month'])

data['day'] = le.fit_transform(data['day'])

data.head()
#train-test splitting

X = data.drop('area', axis = 1)

y = data['area']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,random_state = 0)
#Prediction

rf = RandomForestRegressor()

rf.fit(X_train,y_train)

pred = rf.predict(X_test)

print("MAE:",mean_absolute_error(y_test,pred))

print("MSE:",mean_squared_error(y_test,pred))