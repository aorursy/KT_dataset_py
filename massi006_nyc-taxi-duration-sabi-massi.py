# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 

from datetime import date, datetime

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df=pd.read_csv('../input/train.csv', index_col='id')

test = pd.read_csv("../input/test.csv", parse_dates=["pickup_datetime"])

test.head(20)
# Data Exploration
len(df.index) == df.index.nunique() # check whether ID is unique
type(test['pickup_datetime'])
df['passenger_count'].value_counts()[0:6].plot(kind='pie', subplots=True, figsize=(10, 10));
df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])

df['dropoff_datetime'] = pd.to_datetime(df['dropoff_datetime'])

test.head(20)

#df['dropoff_dropoff'] = pd.to_datetime(df['dropoff_dropoff'])
NUM_FEATURE=['vendor_id','passenger_count', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']

X=df[NUM_FEATURE]

y=df['trip_duration']

X_test=test[NUM_FEATURE]

X_test.head(10)
def extract_date_features(df, col):

    X = pd.DataFrame()

    """ Extract features from a date. """

    

    X[col + '_year'] = df[col].dt.year

    X[col + '_month'] = df[col].dt.month

    X[col + '_week'] = df[col].dt.week

    X[col + '_dow'] = df[col].dt.dayofweek

    X[col + '_hour'] = df[col].dt.hour

    X[col + '_weekday'] = df[col].dt.weekday

    X[col + '_days_in_month'] = df[col].dt.days_in_month

    return X
test['pickup_datetime'].dt.year
launched_features_pickup=extract_date_features(df, 'pickup_datetime')

launched_features_dropoff=extract_date_features(df, 'dropoff_datetime')

launched_features_pickup_test=extract_date_features(test, 'pickup_datetime')

test.head()
X = pd.concat([X, launched_features_pickup], axis=1)

X['flag'] = np.where(df['store_and_fwd_flag']=='N', 0, 1)

X_test = pd.concat([X_test, launched_features_pickup_test], axis=1)

X_test['flag'] = np.where(test['store_and_fwd_flag']=='N', 0, 1)

X_test.head()
print(X.shape)

print(X_test.shape)
# Split
X_test.info()
#Modeling
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.metrics import mean_squared_log_error
rf= RandomForestRegressor(n_estimators=10, min_samples_leaf=10, min_samples_split=15, max_features='auto')

rf

# min_samples_split=2, min_samples_leaf=4
score=-cross_val_score(rf, X, y, cv=5, scoring='neg_mean_squared_log_error')

score=np.sqrt(score)

score
rf.fit(X,y)
pred=rf.predict(X_test)
recup_sum['id']=X_test.index

X_test.head()

#recup_sum['id']
recup_sum=pd.read_csv("../input/sample_submission.csv")

recup_sum['trip_duration']=pred

recup_sum.head()
recup_sum.to_csv("submition_massi.csv", index=False)