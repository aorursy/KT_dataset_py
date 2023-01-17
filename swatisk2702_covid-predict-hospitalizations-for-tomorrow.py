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
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from datetime import datetime, timedelta

from sklearn import preprocessing

import xgboost as xgb

import matplotlib.pyplot as plt

from xgboost import plot_importance

from sklearn.metrics import mean_squared_error

from sklearn.utils import shuffle

from sklearn.model_selection import train_test_split

import pickle
file = "../input/uncover/RDSC-07-30-Update/RDSC-07-30-Update/coronadatascraper/coronadatascraper-timeseries.csv"

covid_stats = pd.read_csv(file)

covid_stats.head()

X_columns = ['state', 'country',

       'population', 'lat', 'long', 'cases',

       'deaths', 'recovered', 'active', 'tested',

       'hospitalized_current', 'icu', 'icu_current',

       'growthfactor', 'date']



covid_stats['date'] = pd.to_datetime(covid_stats['date'])



covid_stats_X = covid_stats[X_columns]
df1 = covid_stats_X.loc[

    (covid_stats_X['hospitalized_current'].notnull())

]
id_cols = ['date','country','state','population']



def lag_feature(df, lags, col):

    tmp = df[id_cols + [col]]

    for i in lags:

        shifted = tmp.copy()

        shifted.columns = id_cols + [(col+'_lag_'+str(i))]

        shifted['date'] += timedelta(days=i)

        df = pd.merge(df, shifted, on=id_cols, how='left')

    return df
cases_to_consider = ['hospitalized_current','deaths','tested']



for c in cases_to_consider:

    if c == 'hospitalized_current':

        df1 = lag_feature(df1,[1,3,7],c)

    if c == 'deaths':

        df1 = lag_feature(df1,[1],c)

    if c == 'tested':

        df1 = lag_feature(df1,[7],c)
df1.head()
df1['state'] = df1['state'].astype(str)



LE = preprocessing.LabelEncoder()



df1['state'] = LE.fit_transform(df1['state'])

df1['country'] = LE.fit_transform(df1['country'])

model = xgb.XGBRegressor(max_depth=8,n_estimators=1000,

                     min_child_weight=300,colsample_bytree=0.8,

                     subsample=0.8,eta=0.3,seed=42)
X = df1



X = shuffle(X)



Y = X['hospitalized_current']



X.drop(['hospitalized_current'],axis = 1,inplace=True)

X.drop(['cases'],axis = 1,inplace=True)

X.drop(['deaths'],axis = 1,inplace=True)

X.drop(['recovered'],axis = 1,inplace=True)

X.drop(['tested'],axis = 1,inplace=True)

X.drop(['date'],axis = 1,inplace=True)



X_train, X_valid, y_train, y_valid = train_test_split(X, Y, test_size=0.30, random_state=42)



X_valid_a, X_test, y_valid_a, y_test = train_test_split(X_valid,y_valid,test_size = .10,random_state=42)
model.fit(X_train,y_train,eval_metric="rmse",

          eval_set=[(X_train, y_train), (X_valid_a, y_valid_a)],

          verbose=True,early_stopping_rounds = 10)
def plot_features(booster, figsize):    

    fig, ax = plt.subplots(1,1,figsize=figsize)

    return plot_importance(booster=booster, ax=ax)



plot_features(model, (10,14))
y_test_pred = model.predict(X_test)

mean_squared_error(y_test, y_test_pred)
filehandler = open('object_model_1.md', 'wb') 

pickle.dump(model, filehandler)