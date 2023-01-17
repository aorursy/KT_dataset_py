import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os



print(os.listdir("../input"))

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import datetime

from sklearn import datasets, linear_model

from sklearn.metrics import mean_squared_error

%matplotlib inline



data = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

test.dtypes



print(test.head(5))

data.dtypes
# Extract hours from datetime

data['datetime'] = pd.to_datetime(data['datetime'])

data['hour'] = data['datetime'].dt.hour

data['month'] = data['datetime'].dt.month



test['datetime'] = pd.to_datetime(test['datetime'])

test['hour'] = data['datetime'].dt.hour

test['month'] = data['datetime'].dt.month



data['season'] = data.season.astype('category')

data['month'] = data.month.astype('category')

data['hour'] = data.hour.astype('category')

data['holiday'] = data.holiday.astype('category')

data['workingday'] = data.workingday.astype('category')

data['weather'] = data.weather.astype('category')





test['season'] = test.season.astype('category')

test['month'] = test.month.astype('category')

test['hour'] = test.hour.astype('category')

test['holiday'] = test.holiday.astype('category')

test['workingday'] = test.workingday.astype('category')

test['weather'] = test.weather.astype('category')





data.dtypes
data = data.drop(['atemp', 'casual', 'registered', 'windspeed'], axis=1)

test = test.drop(['atemp','windspeed'], axis=1)
test.head(2)
import math

data['count'] = data['count'].transform(lambda x: math.log(x))
data = data.drop(['datetime'], axis=1)

data_dummy = data



#test = test.drop(['datetime'], axis=1)

test_dummy = test



def dummify_dataset(df, column):       

    df = pd.concat([df, pd.get_dummies(df[column], prefix=column, drop_first=True)],axis=1)

    df = df.drop([column], axis=1)

    return df



columns_to_dummify = ['season', 'month', 'hour', 'holiday', 'workingday', 'weather']

for column in columns_to_dummify:

    data_dummy = dummify_dataset(data_dummy, column)

    test_dummy = dummify_dataset(test_dummy, column)

    





test_dummy.head(5)
from sklearn.model_selection import train_test_split



y = data_dummy['count']

X = data_dummy.drop(['count'], axis=1)



X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33, random_state=42)

from sklearn.tree import DecisionTreeRegressor

from sklearn.linear_model import LinearRegression, Ridge, HuberRegressor, ElasticNetCV

from sklearn.metrics import mean_squared_log_error 

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold

from sklearn.ensemble import BaggingRegressor, ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor

etr = ExtraTreesRegressor(max_depth= 20, n_estimators= 500)

#etr.fit(X_train, y_train)

#Y_to_train = train_sample["count"]

#X_to_train = train_sample.drop(['count'], axis=1)



etr.fit(X_train,y_train)

#y_pred = etr.predict(test_sample)
test_with_datetime = pd.read_csv("../input/test.csv")

test_dummy = test_dummy.drop(['datetime'], axis=1)

test_predictions = etr.predict(test_dummy)
np.exp(test_predictions )
predictions =  np.exp(test_predictions )

submission = pd.DataFrame({ 'datetime': test.datetime.values, 'count': predictions })

submission.to_csv("my_submission_10.csv", index=False)