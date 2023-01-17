# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import datetime as dt

import datetime

import math

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score, mean_squared_error

from sklearn.linear_model import Ridge

from sklearn import linear_model

from sklearn import svm

from math import sqrt 

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv('../input/data.csv')

del train_data['Total_distance']

fast_data_1 = pd.read_csv('../input/fastest_routes_train_part_1.csv')

fast_data_2 = pd.read_csv('../input/fastest_routes_train_part_2.csv')

weather_data = pd.read_csv('../input/weather_data.csv')

dataFrame1 = pd.DataFrame(fast_data_1)

dataFrame2 = pd.DataFrame(fast_data_2)

dataFrame1.append(dataFrame2)

train_data = pd.merge(train_data,dataFrame1,on="ID",how="left")

train_data = pd.merge(train_data,weather_data,on="Date",how="left")

data = pd.DataFrame(train_data)

data = data.dropna(axis=0, how='any')

print(data.columns)
def calculateCorelation(data):

    corr = data.corr(method = 'pearson')

    sns.heatmap(corr, annot=True, fmt = ".2f")
data['Origin_start_time'] = pd.to_datetime(data['Origin_start_time'])

data['Origin_start_time'] = data['Origin_start_time'].dt.hour

data['Destination_end_time'] = pd.to_datetime(data['Destination_end_time'])

data['Destination_end_time'] = data['Destination_end_time'].dt.hour

print(data.head(1))

def haversine_array(lat1, lng1, lat2, lng2):

    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))

    AVG_EARTH_RADIUS = 6371  # in km

    lat = lat2 - lat1

    lng = lng2 - lng1

    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2

    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))

    return h

data.loc[:, 'distance_haversine'] = haversine_array(data['Origin_start_lat'].values, data['Origin_start_lng'].values, data['Destination_end_lat'].values, data['Destination_end_lng'].values)

data.loc[:, 'Speed'] = 1000 * data['distance_haversine'] / data['Total_time']

data = data.dropna(axis=0, how='any')
def selector1(data):

    input_data = pd.DataFrame(data, columns =  ['Origin_start_time','Destination_end_time','Total_distance','Temperature','Humidity'])

    return input_data
def selector2(data):

    input_data = pd.DataFrame(data, columns =  ['Origin_start_lng','Origin_start_lat','Destination_end_lng','Destination_end_lat','Total_distance','Temperature','Humidity'])

    return input_data
def selector3(data):

    input_data = pd.DataFrame(data, columns =  ['Origin_start_lng','Origin_start_lat','Destination_end_lng','Destination_end_lat','Origin_start_time','Destination_end_time','Temperature','Humidity'])

    return input_data
def selector5(data):

    input_data = pd.DataFrame(data, columns =  ['Origin_start_lng','Origin_start_lat','Destination_end_lng','Destination_end_lat','Origin_start_time','Destination_end_time','Total_distance'])

    return input_data
def selector6(data):

    input_data = pd.DataFrame(data, columns =  ['Origin_start_lng','Origin_start_lat','Destination_end_lng','Destination_end_lat','Origin_start_time','Destination_end_time','Total_distance','Temperature','Humidity'])

    return input_data
def training(train_data,label,alg):

    X = np.array(train_data)

    y = np.array(label)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    alg.fit(X_train, y_train)

    prediction = alg.predict(X_test)

    r2 = r2_score(y_test, prediction)

    MSE = mean_squared_error(y_test,prediction)

    RMSE = sqrt(MSE/60)

    print('R2 score:', r2)

    print('MSE:', MSE)

    print('RMSE:',RMSE)

    
calculateCorelation(data)
calculateCorelation(pd.DataFrame(data, columns=['Temperature','Humidity','Total_time']))
calculateCorelation(pd.DataFrame(data, columns=['Total_distance','Total_time']))
linear_regressor = linear_model.LinearRegression()

training(selector1(data),data['Total_time'],linear_regressor)
linear_regressor = linear_model.LinearRegression()

training(selector2(data),data['Total_time'],linear_regressor)
linear_regressor = linear_model.LinearRegression()

training(selector3(data),data['Total_time'],linear_regressor)
linear_regressor = linear_model.LinearRegression()

training(selector5(data),data['Total_time'],linear_regressor)
linear_regressor = linear_model.LinearRegression()

training(selector6(data),data['Total_time'],linear_regressor)
random_forest = RandomForestRegressor(max_depth = 2, random_state=0)

training(selector1(data),data['Total_time'],random_forest)
training(selector2(data),data['Total_time'],random_forest)
training(selector3(data),data['Total_time'],random_forest)
training(selector5(data),data['Total_time'],random_forest)
training(selector6(data),data['Total_time'],random_forest)
gradient_boosting = GradientBoostingRegressor(alpha=0.6,criterion='friedman_mse',

                                             init=None, learning_rate = 0.1, loss='ls', max_depth = 3,

                                             max_features=None, max_leaf_nodes = None, min_samples_leaf=1,

                                             min_samples_split=2, min_weight_fraction_leaf=0.0,

                                             n_estimators=100, presort='auto', random_state=None,

                                             subsample=1.0, verbose=0, warm_start=False)

training(selector1(data),data['Total_time'],gradient_boosting)
training(selector2(data),data['Total_time'],gradient_boosting)
training(selector3(data),data['Total_time'],gradient_boosting)
training(selector5(data),data['Total_time'],gradient_boosting)
training(selector6(data),data['Total_time'],gradient_boosting)
ridge = Ridge(alpha=0.98, copy_X=True, fit_intercept=True,

             max_iter=None, normalize=False, random_state=None,solver='auto',

             tol=0.001)

training(selector1(data),data['Total_time'],ridge)
training(selector2(data),data['Total_time'],ridge)
training(selector3(data),data['Total_time'],ridge)
training(selector5(data),data['Total_time'],ridge)
training(selector6(data),data['Total_time'],ridge)