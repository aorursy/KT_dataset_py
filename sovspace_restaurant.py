import numpy as np
import pandas as pd
import datetime
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from math import * 
df_data = pd.read_csv('../input/restaurants/data.csv')
df_train = pd.read_csv('../input/restaurants/train.csv')
df_test = pd.read_csv('../input/restaurants/test.csv')
df_city_data = pd.read_csv('../input/restaurants/restaurants.csv')
df_data['Date'] = pd.to_datetime(df_data['Date'])
df_train['Date'] = pd.to_datetime(df_train['Date'])
df_test['Date'] = pd.to_datetime(df_test['Date'])
df_data.info()
df_cleaned_data = df_data[df_data['Date'] < pd.to_datetime(('2013-01-01'))]
df_cleaned_data = df_cleaned_data.sort_values(by=['Date'])
df_train = df_train.sort_values(by=['Date'])
df_train.info()
df_data.info()
df_city_data = df_city_data.sort_values(by=['City'])
df_city_data.info()
df_merged = pd.merge_asof(df_train, df_cleaned_data,
                      on='Date',
                      by='City')
df_merged['IsHoliday_y'].equals(df_merged['IsHoliday_x'])
df_merged = df_merged.drop(columns=['IsHoliday_y'])
df_merged = df_merged.rename(columns = {'IsHoliday_x' : 'IsHoliday'})
df_merged = df_merged.sort_values(by=['City'])
df_merged.head()
df_full_train_data = pd.merge_asof(df_merged, df_city_data, on = 'City')
df_full_train_data['IsHoliday'] = df_full_train_data['IsHoliday'].astype(int) 
df_full_train_data.head()
df_full_train_data.to_csv('full_train_data.csv')
df_full_train_data.boxplot(['Weekly_Sales'])
df_full_train_data.hist(['Weekly_Sales'], bins=100)
df_full_train_data.boxplot('Temperature')
df_full_train_data.boxplot('Fuel_Price')
train = df_full_train_data.copy()
train['Date'] =  train["Date"].dt.dayofyear
train['Unemployment'] = train['Unemployment'] * train['Size'] / 100
train.mean()
train = train[(train['Weekly_Sales'] < train['Weekly_Sales'].quantile(0.75))]
train = train[(train['Temperature'] > train['Temperature'].quantile(0.05))]
y = train.pop('Weekly_Sales').values
X = train.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', learning_rate = 0.15, max_depth = 12, alpha=12, n_estimators = 400, verbosity=1, 
                          grow_policy = 'lossguide', max_leaves=128,)
xg_reg.fit(X_train, y_train, eval_metric="rmse", eval_set=[(X_test, y_test)], verbose=True)
mse = mean_squared_error(y_test, xg_reg.predict(X_test))

print("RMSE: %.4f" % sqrt(mse))
mse = mean_squared_error(y_train, xg_reg.predict(X_train))
print("RMSE: %.4f" % sqrt(mse))
print(xg_reg.predict(X_test[0:10]))
print(y_test[0:10])
df_test = df_test.sort_values(by='Date')
df_merged_test = pd.merge_asof(df_test, df_cleaned_data,
                      on='Date',
                      by='City')
df_merged_test['IsHoliday_y'].equals(df_merged_test['IsHoliday_x'])
df_merged_test = df_merged_test.drop(columns=['IsHoliday_y'])
df_merged_test = df_merged_test.rename(columns = {'IsHoliday_x' : 'IsHoliday'})
df_merged_test = df_merged_test.sort_values(by=['City'])
df_merged_test = pd.merge_asof(df_merged_test, df_city_data, on = 'City')
df_merged_test['IsHoliday'] = df_full_train_data['IsHoliday'].astype(int) 
test = df_merged_test.copy()
test['Date'] =  test["Date"].dt.dayofyear
test['Unemployment'] = test['Unemployment'] * test['Size'] / 100
test["Weekly sales"] = xg_reg.predict(test.to_numpy())
test.to_csv('results.csv')