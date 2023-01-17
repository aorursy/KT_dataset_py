import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
my_file = pd.read_excel('/kaggle/input/flight-fare-prediction-mh/Data_Train.xlsx')
my_file.head(10)
dataset = my_file.sort_values(by=['Date_of_Journey'])
dataset.head(3)
dataset.isna().any()
#Checking for null values in Route and Total_Stops column
null_in_route = dataset[dataset['Route'].isnull()]
null_in_total_stops = dataset[dataset['Total_Stops'].isnull()]
null_in_route
null_in_total_stops
#Delete the above row by using the index value
delete_row = dataset[dataset['Total_Stops'].isnull()].index
print(delete_row)
dataset = dataset.drop(delete_row)
#Checking for null values
print(dataset.isnull().values.any())
#Get the total rows and column in the dataset
print("Train file shape", dataset.shape)
#Checking the types of column
print(dataset.dtypes)
#Convert Date_of_Journey, Arrival_Time to datetime format
dataset['Date_of_Journey'] = pd.to_datetime(dataset['Date_of_Journey'])
dataset['Arrival_Time'] = pd.to_datetime(dataset['Arrival_Time'])
#Checking the 'Duration' column to see if there is any row which takes only minutes to travel. This may be an outlier
def check_duration_col(row):
    if 'h' not in row:
        print(row)
dataset['Duration'].apply(check_duration_col)
#Since duration of travel cannot be 5 mins hence removing the row.
dataset = dataset.drop(dataset[dataset['Duration'] == '5m'].index)
#Converting Duration column from hours and minutes to minutes only
import re
def convert_hours_to_mins(row):
    row = row.split(' ')
    match_hr = re.match(r"([0-9]+)([a-z]+)", row[0], re.I)
    if len(row) > 1:
        match_min = re.match(r"([0-9]+)([a-z]+)", row[1], re.I)
        mins = match_min.groups()
    else:
        mins = [0]
    if match_hr:
        hr = match_hr.groups()
    hr_to_min = int(hr[0]) * 60
    return int(hr_to_min) + int(mins[0])
dataset['Duration'] = dataset['Duration'].apply(convert_hours_to_mins)
dataset.head(3)
#Remove stops from Total_stops
def remove_stops(row):
    if row == 'non-stop':
        row = 0
    else:
        row = row.split(' ')[0]
    return row
dataset['Total_Stops'] = dataset['Total_Stops'].apply(remove_stops)
dataset['Additional_Info'] = dataset['Additional_Info'].str.lower()
dataset['Additional_Info'].value_counts()
#Remove colon from Departure Time
def remove_colon(row):
    if ':' in str(row):
        row = row.replace(':', '')
    return int(row)

dataset['Dep_Time'] = dataset['Dep_Time'].apply(remove_colon)
#Selecting features from dataframe to prepare data for training
dataframe_x = dataset[['Airline', 'Source', 'Destination', 'Dep_Time', 'Duration', 'Total_Stops', 'Additional_Info']]
dataframe_y = dataset[['Price']]
#Converting categorical columns to one-hot encoding 
dataframe_x = pd.get_dummies(dataframe_x, columns=['Source', 'Additional_Info', 'Airline', 'Destination'])
dataframe_x.head(5)
#Splitting the data into train, validation and test 
train_dataframe_x = dataframe_x[:7000]
train_dataframe_y = dataframe_y[:7000]
val_dataframe_x = dataframe_x[7000:8500]
val_dataframe_y = dataframe_y[7000:8500]
test_dataframe_x = dataframe_x[8500:]
test_dataframe_y = dataframe_y[8500:]
print(train_dataframe_x.shape, val_dataframe_x.shape, test_dataframe_x.shape)
print(train_dataframe_y.shape, val_dataframe_y.shape, test_dataframe_y.shape)
#Standardizing the datasets
scaler = preprocessing.StandardScaler()
train_dataframe_x = scaler.fit_transform(train_dataframe_x)
val_dataframe_x = scaler.transform(val_dataframe_x)
test_dataframe_x = scaler.transform(test_dataframe_x)
#Finding the best hyperparameters to train the Random Forest model
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import datetime
param_grid = {
 'max_depth': [4, 8, 16, 32],
 'n_estimators': [1, 2, 5, 10, 50, 100, 200]
}
t1 = datetime.datetime.now()
rf = RandomForestRegressor(n_jobs=-1)
clf = GridSearchCV(estimator = rf, param_grid = param_grid)
clf.fit(val_dataframe_x,val_dataframe_y)
print("time required = ", datetime.datetime.now() - t1)
clf.best_params_
rf_model = RandomForestRegressor(max_depth = clf.best_params_['max_depth'], n_estimators=clf.best_params_['n_estimators'])
rf_model.fit(train_dataframe_x, train_dataframe_y)
#Getting scores
print("Train Score", rf_model.score(train_dataframe_x, train_dataframe_y))
print("Test Score", rf_model.score(test_dataframe_x, test_dataframe_y))
#making prediction on test data
y_pred = rf_model.predict(test_dataframe_x)
#Calculating the metrics
from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(test_dataframe_y, y_pred))
print('MSE:', metrics.mean_squared_error(test_dataframe_y, y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(test_dataframe_y, y_pred)))
print("R-squared score:", metrics.r2_score(test_dataframe_y, y_pred))
#Using XGBoost Regressor
#Finding best parameters
params = {'max_depth': [1, 2, 3, 4, 5], 'learning_rate': [0.01, 0.05, 0.1], 'n_estimators': [20, 50, 100, 200, 300, 400, 500], 'reg_lambda': [0.001, 0.1, 1.0, 10.0, 100.0]}
xgb_model = RandomizedSearchCV(XGBRegressor(), params, n_iter=20, cv=10, n_jobs=-1)
xgb_model.fit(train_dataframe_x, train_dataframe_y)
#Getting scores
print("Train Score", xgb_model.score(train_dataframe_x, train_dataframe_y))
print("Test Score", xgb_model.score(test_dataframe_x, test_dataframe_y))
#making prediction on test data
y_pred_xgb = xgb_model.predict(test_dataframe_x)
#Calculating the metrics
print('MAE:', metrics.mean_absolute_error(test_dataframe_y, y_pred_xgb))
print('MSE:', metrics.mean_squared_error(test_dataframe_y, y_pred_xgb))
print('RMSE:', np.sqrt(metrics.mean_squared_error(test_dataframe_y, y_pred_xgb)))
print("R-squared score:", metrics.r2_score(test_dataframe_y, y_pred_xgb))