import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
from sklearn import tree, ensemble, metrics, linear_model, preprocessing, model_selection, feature_selection

from plotnine import *

import matplotlib.pyplot as plt

import seaborn as sns
# Create class which performs Label Encoding - if required

class categorical_encoder:

    def __init__(self, columns, kind = 'label', fill = True):

        self.kind = kind

        self.columns = columns

        self.fill = fill

        

    def fit(self, X):

        self.dict = {}

        self.fill_value = {}

        

        for col in self.columns:

            label = preprocessing.LabelEncoder().fit(X[col])

            self.dict[col] = label

            

            # To fill

            if self.fill:

                self.fill_value[col] = X[col].mode()[0]

                X[col] = X[col].fillna(self.fill_value[col])

                

        print('Label Encoding Done for {} columns'.format(len(self.columns)))

        return self

    def transform(self, X):

        for col in self.columns:

            if self.fill:

                X[col] = X[col].fillna(self.fill_value[col])

                

            X.loc[:, col] = self.dict[col].transform(X[col])

        print('Transformation Done')

        return X

    

def categorical(df):

    df['var2'] = df['var2'].replace({'A': 1, 'B': 2, 'C': 3})

    return df



test = pd.read_csv('/kaggle/input/test_pavJagI.csv')

train = pd.read_csv('/kaggle/input/train_6BJx641.csv')



train['datetime'] = pd.to_datetime(train['datetime'], yearfirst = True)

test['datetime'] = pd.to_datetime(test['datetime'], yearfirst = True)
# The following approach of merging train and test works for competitions but not during production because in production 

# it is required to replicate the operations performed on the training set on the test set. However, if done correctly, the 

# results will be similar in the approaches.



# Indicators

train['which'] = 1

test['which'] = 0



# Merge

data = pd.concat([train, test], axis = 0, ignore_index = True)

data = data.set_index('datetime')

data = data.sort_index()



############  Create New Features - with different lags ################

data['temperature_rolling'] = data['temperature'].rolling('24H').mean()

data['var1_rolling'] = data['var1'].rolling('24H').mean()

data['windspeed_rolling'] = data['windspeed'].rolling('24H').mean()



# Week lagged features

data['temperature_rolling_week'] = data['temperature'].rolling('168H').mean()

data['var1_rolling_week'] = data['var1'].rolling('168H').mean()

data['windspeed_rolling_week'] = data['windspeed'].rolling('168H').mean()



# Other Lags

data['temperature_rolling_3d'] = data['temperature'].rolling('72H').mean()

data['temperature_rolling_6h'] = data['temperature'].rolling('6H').mean()

data = data.reset_index()





# y lagged values

temp_med = data.set_index('datetime').resample('2M')['electricity_consumption'].mean()

temp_med.name = 'y_med_month'

data = pd.concat([data.set_index('datetime'),temp_med], 

          axis = 1).fillna(method = 'bfill').reset_index()



# Split Back

train = data.loc[data['which'] == 1, :].drop('which', axis = 1)

test = data.loc[data['which'] == 0, :].drop(['which', 'electricity_consumption'], axis = 1)
# Create Time Series datetime features

def ts_features(df, col = 'datetime'):

    #df['dayofmonth'] = df[col].dt.day

    df['weekday'] = df[col].dt.dayofweek

    df['weekend'] = (df[col].dt.dayofweek >= 5)*1

    df['month'] = df[col].dt.month

    df['hour'] = df[col].dt.hour

    df['year'] = df[col].dt.year

    return df
X_train = ts_features(train).drop(['datetime', 'electricity_consumption', 'ID'], axis = 1)

y_train = ts_features(train)['electricity_consumption']



X_test = ts_features(test).drop(['ID', 'datetime'], axis = 1)



test_id = test.ID

train_id = train.ID



X_train = categorical(X_train)

X_test = categorical(X_test)
X_train.head()
X_test = X_test.fillna(method = 'ffill')
import lightgbm as lgb

model = lgb.LGBMRegressor(n_estimators = 4000, learning_rate = .02, 

                          max_features = .7, max_depth = 3, subsample = .9).fit(X_train, y_train)

# Make Submission

submission = pd.DataFrame()

submission['ID'] = test_id



# Take a weighted average of the predictions

submission['electricity_consumption'] = model.predict(X_test)



# Save submission file

submission.to_csv('/kaggle/working/submission.csv', index = None)
from sklearn import pipeline

import xgboost as xgb

import catboost as cb



model = ensemble.RandomForestRegressor(n_estimators = 250, max_depth = 20, min_samples_leaf = 5,

                                       n_jobs = 4).fit(X_train, y_train)

model1 = lgb.LGBMRegressor(n_estimators = 4000, learning_rate = .02, max_features = .7, max_depth = 3, subsample = .9).fit(X_train, y_train)

model2 = xgb.XGBRegressor(n_estimators = 2000, learning_rate = .04, max_features = .7, max_depth = 3, subsample = .9).fit(X_train, y_train)

model4 = cb.CatBoostRegressor(n_estimators = 2000, learning_rate = .04, max_depth = 3,

                              rsm = .7, subsample = .9, silent = True).fit(X_train, y_train)
submission = pd.DataFrame()

submission['ID'] = test_id



# Take a weighted average of the predictions

submission['electricity_consumption'] = (model.predict(X_test)+(3*model1.predict(X_test)+(2*model2.predict(X_test))+\

                                                               (3*model4.predict(X_test))))/9

submission.to_csv('/kaggle/working/submission.csv', index = None)