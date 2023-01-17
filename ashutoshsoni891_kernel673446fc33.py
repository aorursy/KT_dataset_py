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
train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/train.csv' , parse_dates = True)
test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/test.csv')
sub = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/submission.csv')

train[:5]
test[:5]
sub[:5]
train['Date'] = pd.to_datetime(train['Date'])
train['day'] = train['Date'].dt.day
train['month'] = train['Date'].dt.month
train['year'] = train['Date'].dt.year
train['day_of_year'] = train['Date'].dt.dayofyear
train['dayofyear'] = train['Date'].dt.dayofyear
train['weekofyear'] = train['Date'].dt.weekofyear
train['week'] = train['Date'].dt.week
train['dayofweek'] = train['Date'].dt.dayofweek
train['weekday'] = train['Date'].dt.weekday
train['quarter'] = train['Date'].dt.quarter
train['daysinmonth'] = train['Date'].dt.daysinmonth
train['is_month_start'] = train['Date'].dt.is_month_start
train['is_month_end'] = train['Date'].dt.is_month_end
train['is_quarter_start'] = train['Date'].dt.is_quarter_start
train['is_quarter_end'] = train['Date'].dt.is_quarter_end
train['is_year_start'] = train['Date'].dt.is_year_start
train['is_year_end'] = train['Date'].dt.is_year_end
train['is_leap_year'] = train['Date'].dt.is_leap_year

train.shape
train.info()
def train_test_split(df , per =0.7):
    split = int(per * len(df))
    trainX = df.drop(['Id' , 'Date' , 'Province/State' , 'Country/Region' ,'ConfirmedCases','Fatalities' ,'year', 'quarter' , 'is_quarter_start' , 'is_quarter_end', 'is_year_end' , 'is_leap_year'] , axis = 1)[:split]
    trainY1 = df['ConfirmedCases'][:split]
    trainY2 = df['Fatalities'][:split]
    testX = df.drop(['Id' , 'Date' , 'Province/State' , 'Country/Region' ,'ConfirmedCases','Fatalities','year', 'quarter' , 'is_quarter_start' , 'is_quarter_end', 'is_year_end' , 'is_leap_year'] , axis = 1)[:-split]
    testY1 = df['ConfirmedCases'][:-split]
    testY2 = df['Fatalities'][:-split]
    trainX = trainX.replace(True , 1).replace(False , 0)
    testX = testX.replace(True , 1).replace(False , 0)
    
    return trainX ,trainY1 , trainY2 ,testX , testY1 , testY2



    
trainX ,trainY1 , trainY2 ,testX , testY1 , testY2 = train_test_split(train)
print(trainX.shape ,trainY1.shape , trainY2.shape ,testX.shape , testY1.shape , testY2.shape)
for cols in trainX.columns:
    print('FOR COLUMN :' , cols)
    print(train[cols].value_counts())
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor

xgb1 = xgb.XGBRegressor()
xgb1.fit(trainX , trainY2)

xgb2 = xgb.XGBRegressor()
xgb2.fit(trainX , trainY1)

rf1 = RandomForestRegressor()
rf1.fit(trainX , trainY2)

rf2 = RandomForestRegressor()
rf2.fit(trainX , trainY1)

from sklearn.metrics import mean_squared_error

xgb1_predict = xgb1.predict(testX)
print('xgb1_predict : ', mean_squared_error(xgb1_predict , testY2))

xgb2_predict = xgb1.predict(testX)
print('xgb2_predict',mean_squared_error(xgb2_predict , testY1))


rf1_predict = rf1.predict(testX)
print('rf1_predict : ', mean_squared_error(rf1_predict , testY2))

rf2_predict = rf1.predict(testX)
print('rf2_predict',mean_squared_error(rf2_predict , testY1))


import matplotlib.pyplot as plt

plt.scatter(rf1_predict , testY2 )
plt.scatter(rf2_predict , testY1)


test['Date'] = pd.to_datetime(test['Date'])


test['day'] = test['Date'].dt.day
test['month'] = test['Date'].dt.month
test['year'] = test['Date'].dt.year
test['day_of_year'] = test['Date'].dt.dayofyear
test['dayofyear'] = test['Date'].dt.dayofyear
test['weekofyear'] = test['Date'].dt.weekofyear
test['week'] = test['Date'].dt.week
test['dayofweek'] = test['Date'].dt.dayofweek
test['weekday'] = test['Date'].dt.weekday
test['quarter'] = test['Date'].dt.quarter
test['daysinmonth'] = test['Date'].dt.daysinmonth
test['is_month_start'] = test['Date'].dt.is_month_start
test['is_month_end'] = test['Date'].dt.is_month_end
test['is_quarter_start'] = test['Date'].dt.is_quarter_start
test['is_quarter_end'] = test['Date'].dt.is_quarter_end
test['is_year_start'] = test['Date'].dt.is_year_start
test['is_year_end'] = test['Date'].dt.is_year_end
test['is_leap_year'] = test['Date'].dt.is_leap_year




test = test.drop([ 'Date' , 'Province/State' , 'Country/Region'  ,'year', 'quarter' , 'is_quarter_start' , 'is_quarter_end', 'is_year_end' , 'is_leap_year'], axis = 1)
test[:5]

test = test.drop('ForecastId' , axis = 1)
test[:5]
Fatalities = rf1.predict(test)
ConfirmedCases = rf2.predict(test)
submission  = pd.DataFrame({'ForecastId' : test['ForecastId'],
             'ConfirmedCases': ConfirmedCases,
             'Fatalities' : Fatalities})
submission.to_csv('submission.csv' , index = False)

