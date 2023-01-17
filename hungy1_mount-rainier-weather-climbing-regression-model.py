#importing the libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
#importing the climbing data

df_climbing = pd.read_csv("../input/mount-rainier-weather-and-climbing-data/climbing_statistics.csv")

print(df_climbing.shape)

df_climbing.head()
#importing the weather data

df_weather = pd.read_csv("../input/mount-rainier-weather-and-climbing-data/Rainier_Weather.csv")

print(df_weather.shape)

df_weather.head()
#Merging two datsets, merge should be done based on the feature 'Date'

df =df_weather.merge(df_climbing ,on ='Date')

print(df.shape)

df.head()
#ploting the corolation matrix for dectecting muticolinearity

plt.figure(figsize =(12,7)) 

sns.heatmap(df.corr(), annot =True , cmap ='RdYlGn')
df['Date'].dtype
#'Date' is object we need to convert it to time stamp

df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.month
df.head()
df.info()
#checking for null values

df.isnull().sum()
dataset =df.drop('Date', axis =1)
##split the datset into train and test sets

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(dataset, dataset['Success Percentage'], test_size = 0.2, random_state =0)
#finding the categorical variables

[col for col in X_train.columns if X_train[col].dtype =='O']
#encoding categoricla variable

def categorical_encoding(var, target):

    order = X_train.groupby(var)[target].mean().to_dict()

    X_train[var] = X_train[var].map(order)

    X_test[var]= X_test[var].map(order)
categorical_encoding('Route', 'Success Percentage')
[col for col in X_train.columns if X_train[col].isnull().sum() > 0]
[col for col in X_test.columns if X_test[col].isnull().sum() > 0]
X_train = X_train.drop('Success Percentage', axis =1)

X_test= X_test.drop('Success Percentage', axis =1)
X_train.describe()


from sklearn.metrics import mean_squared_error
import xgboost as xgb

xgb_model = xgb.XGBRegressor()



xgb_model.fit(X_train, y_train)

y_pred = xgb_model.predict(X_train)

print('training mse :' ,mean_squared_error(y_train, y_pred))



y_pred = xgb_model.predict(X_test)

print('test mse : ' , mean_squared_error(y_test, y_pred))
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import GridSearchCV





rf = RandomForestRegressor()

rf.fit(X_train, y_train)



y_pred_rf_train = rf.predict(X_train)

y_pred_rf_test = rf.predict(X_test)



print('rf train mse: {}'.format(mean_squared_error(y_train, y_pred_rf_train)))

print('rf test mse: {}'.format(mean_squared_error(y_test, y_pred_rf_test)))