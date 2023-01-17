# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import datetime as dt





from sklearn.model_selection import train_test_split

import xgboost as xgb



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv", nrows = 2000000)

test = pd.read_csv("../input/test.csv",nrows = 2000000)
train.dtypes
train.isnull().sum()
test.isnull().sum()
train.describe()
train = train.loc[(train['fare_amount'] > 0) & (train['fare_amount'] < 200)]

train = train.loc[(train['pickup_longitude'] > -300) & (train['pickup_longitude'] < 300)]

train = train.loc[(train['pickup_latitude'] > -300) & (train['pickup_latitude'] < 300)]

train = train.loc[(train['dropoff_longitude'] > -300) & (train['dropoff_longitude'] < 300)]

train = train.loc[(train['dropoff_latitude'] > -300) & (train['dropoff_latitude'] < 300)]

#train = train.loc[train[columns_to_select] < ]

# Let's assume taxa's can be mini-busses as well, so we select up to 8 passengers.

train = train.loc[train['passenger_count'] <= 8]

train.describe()
combine = [test, train]

for dataset in combine:

    # Distance is expected to have an impact on the fare

    dataset['longitude_distance'] = dataset['pickup_longitude'] - dataset['dropoff_longitude']

    dataset['latitude_distance'] = dataset['pickup_latitude'] - dataset['dropoff_latitude']

    

    # Straight distance

    dataset['distance_travelled'] = (dataset['longitude_distance'] ** 2 + dataset['latitude_distance'] ** 2) ** .5

    dataset['distance_travelled_sin'] = np.sin((dataset['longitude_distance'] ** 2 * dataset['latitude_distance'] ** 2) ** .5)

    dataset['distance_travelled_cos'] = np.cos((dataset['longitude_distance'] ** 2 * dataset['latitude_distance'] ** 2) ** .5)

    dataset['distance_travelled_sin_sqrd'] = np.sin((dataset['longitude_distance'] ** 2 * dataset['latitude_distance'] ** 2) ** .5) ** 2

    dataset['distance_travelled_cos_sqrd'] = np.cos((dataset['longitude_distance'] ** 2 * dataset['latitude_distance'] ** 2) ** .5) ** 2

    

    # Haversine formula for distance

    # Haversine formula:	a = sin²(Δφ/2) + cos φ1 ⋅ cos φ2 ⋅ sin²(Δλ/2)

    R = 6371e3 # Metres

    phi1 = np.radians(dataset['pickup_latitude'])

    phi2 = np.radians(dataset['dropoff_latitude'])

    phi_chg = np.radians(dataset['pickup_latitude'] - dataset['dropoff_latitude'])

    delta_chg = np.radians(dataset['pickup_longitude'] - dataset['dropoff_longitude'])

    a = np.sin(phi_chg / 2) ** .5 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_chg / 2) ** .5

    c = 2 * np.arctan2(a ** .5, (1-a) ** .5)

    d = R * c

    dataset['haversine'] = d

    

    # Bearing

    # Formula:	θ = atan2( sin Δλ ⋅ cos φ2 , cos φ1 ⋅ sin φ2 − sin φ1 ⋅ cos φ2 ⋅ cos Δλ )

    y = np.sin(delta_chg * np.cos(phi2))

    x = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(delta_chg)

    dataset['bearing'] = np.arctan2(y, x)

    

    # Maybe time of day matters? Obviously duration is a factor, but there is no data for time arrival

    # Features: hour of day (night vs day), month (some months may be in higher demand) 

    dataset['pickup_datetime'] = pd.to_datetime(dataset['pickup_datetime'])

    dataset['hour_of_day'] = dataset.pickup_datetime.dt.hour

    dataset['day'] = dataset.pickup_datetime.dt.day

    dataset['week'] = dataset.pickup_datetime.dt.week

    dataset['month'] = dataset.pickup_datetime.dt.month

    dataset['day_of_year'] = dataset.pickup_datetime.dt.dayofyear

    dataset['week_of_year'] = dataset.pickup_datetime.dt.weekofyear
train = train.loc[train['haversine'] != 0]

train = train.dropna()

train.head()
train.isnull().sum()
test.isnull().sum()
median = test['haversine'].median()

test['haversine'] = test['haversine'].fillna(median)
colormap = plt.cm.RdBu

plt.figure(figsize=(20,20))

plt.title('Pearson Correlation of Features', y=1.05, size=15)

sns.heatmap(train.corr(),linewidths=0.1,vmax=1.0, 

            square=True, cmap=colormap, linecolor='white', annot=True)
train_features_to_keep = ['haversine', 'fare_amount']

train.drop(train.columns.difference(train_features_to_keep), 1, inplace=True)



test_features_to_keep = ['haversine', 'key']

test.drop(test.columns.difference(test_features_to_keep), 1, inplace=True)
x_train = train.drop('fare_amount', axis=1)

y_train = train['fare_amount']

x_test = test.drop('key', axis=1)



# Set up the models.

# Linear Regression Model

from sklearn.linear_model import LinearRegression

regr = LinearRegression()

regr.fit(x_train, y_train)

regr_pred = regr.predict(x_test)



# Random Forest Regressor

from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor()

rfr.fit(x_train, y_train)

rfr_pred = rfr.predict(x_test)
x_pred = test.drop('key', axis=1)



# Let's run XGBoost and predict those fares!

x_train,x_test,y_train,y_test = train_test_split(train.drop('fare_amount',axis=1),train.pop('fare_amount'),random_state=123,test_size=0.2)



def XGBmodel(x_train,x_test,y_train,y_test):

    matrix_train = xgb.DMatrix(x_train,label=y_train)

    matrix_test = xgb.DMatrix(x_test,label=y_test)

    model=xgb.train(params={'objective':'reg:linear','eval_metric':'rmse'}

                    ,dtrain=matrix_train,num_boost_round=200, 

                    early_stopping_rounds=20,evals=[(matrix_test,'test')],)

    return model



model=XGBmodel(x_train,x_test,y_train,y_test)

xgb_pred = model.predict(xgb.DMatrix(x_pred), ntree_limit = model.best_ntree_limit)
regr_pred, rfr_pred, xgb_pred
regr_weight = 1

rfr_weight = 1

xgb_weight = 3

prediction = (regr_pred * regr_weight + rfr_pred * rfr_weight + xgb_pred * xgb_weight) / (regr_weight + rfr_weight + xgb_weight)
prediction
submission = pd.DataFrame({

        "key": test['key'],

        "fare_amount": prediction.round(2)

})



submission.to_csv('sub_fare.csv',index=False)
submission