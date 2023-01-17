# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import os



import matplotlib.pyplot as plt

import numpy as np

import pandas as pd 

import seaborn as sns 

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

from sklearn.model_selection import cross_val_score

from sklearn.tree import DecisionTreeClassifier

import scipy



%matplotlib inline
train = pd.read_csv ('../input/train.csv')

test = pd.read_csv ('../input/test.csv')

sample_submission = pd.read_csv ('../input/sample_submission.csv')
train.columns
test.columns
train.shape
test.shape
train.head()
train.tail()
test.head()
test.tail()
train.info()
test.info()
train.describe()
train.select_dtypes
train.dtypes
test.dtypes
train.trip_duration.min()
train.trip_duration.max()
train["pickup_datetime"].head()
train["dropoff_datetime"].head()
train["store_and_fwd_flag"].head()
plt.subplots(figsize=(18,7))

plt.title("outliers repartition")

train.boxplot()
train.loc[train.trip_duration<5000,"trip_duration"].hist(bins=200,

                                                        color= "#FF1493"

                                                        )
train['log_trip_duration'] = np.log(train['trip_duration'])

plt.hist(train['log_trip_duration'].values, bins=150,color= "#A569BD" )

plt.xlabel('log(trip_duration)')

plt.ylabel('number of train records')

plt.show()
import matplotlib.pyplot as plt 

from matplotlib import animation

from matplotlib import cm

import base64

import io

longitude = list(train.pickup_longitude) + list(train.dropoff_longitude)

latitude = list(train.pickup_latitude) + list(train.dropoff_latitude)

plt.figure(figsize = (10, 10))

plt.plot(longitude,latitude,'.', alpha = 1, markersize = 25)

plt.show()
train.isnull().sum()
train.dropna(inplace=True)

train.isnull().sum()
train = train[(train.trip_duration < 5000)]

train.info()
train['pickup_datetime'] = pd.to_datetime(train['pickup_datetime'], format='%Y-%m-%d %H:%M:%S')

test['pickup_datetime'] = pd.to_datetime(test['pickup_datetime'], format='%Y-%m-%d %H:%M:%S')
train['hour'] = train.loc[:,'pickup_datetime'].dt.hour;

train['week'] = train.loc[:,'pickup_datetime'].dt.week;

train['weekday'] = train.loc[:,'pickup_datetime'].dt.weekday;

train['hour'] = train.loc[:,'pickup_datetime'].dt.hour;

train['month'] = train.loc[:,'pickup_datetime'].dt.month;
test['hour'] = test.loc[:,'pickup_datetime'].dt.hour;

test['week'] = test.loc[:,'pickup_datetime'].dt.week;

test['weekday'] = test.loc[:,'pickup_datetime'].dt.weekday;

test['hour'] = test.loc[:,'pickup_datetime'].dt.hour;

test['month'] = test.loc[:,'pickup_datetime'].dt.month;
train.head()
test.head()
train.shape
test.shape
cat_vars = ['store_and_fwd_flag']
for col in cat_vars:

    train[col] = train[col].astype('category').cat.codes

train.head()
for col in cat_vars:

    test[col] = test[col].astype('category').cat.codes

test.head()
train.columns
selection_train = ["passenger_count","store_and_fwd_flag", "pickup_longitude", "pickup_latitude", "dropoff_longitude","dropoff_latitude", "week", "weekday", "hour", "month"]

selection_test = ["passenger_count","store_and_fwd_flag", "pickup_longitude", "pickup_latitude", "dropoff_longitude","dropoff_latitude", "week", "weekday", "hour", "month"]



#selection_train = ["passenger_count","month", "weekday", "hour", "week", "store_and_fwd_flag"]

#selection_test = ["passenger_count", "month", "weekday", "hour", "week", "store_and_fwd_flag"]



y_train = np.log(train["trip_duration"]) # ma target 

X_train = train[selection_train] # ses features

X_test = test[selection_test]

import numpy as np

from sklearn.preprocessing import StandardScaler

X = np.random.randn(10, 3)

scaler = StandardScaler()

scaler.fit(X)
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)
X_train.shape, X_test.shape, y_train.shape
y_train.head(20)
X_train.head(20)
X_test.head(20)
from sklearn.ensemble import RandomForestRegressor
mrf = RandomForestRegressor(n_estimators=20)

mrf.fit(X_train, y_train)
from sklearn.model_selection import cross_val_score
#crossval_scores = cross_val_score(mrf, X_train, y_train, cv=5, scoring='neg_mean_squared_log_error')

#crossval_scores
#for i in range(len(crossval_scores)):

#    crossval_scores[i] = np.sqrt(abs(crossval_scores[i]))

#crossval_scores
y_test_pred = mrf.predict(X_test)

y_test_pred[:5]
submission = pd.read_csv('../input/sample_submission.csv')

submission.head()
submission['trip_duration']= np.exp(y_test_pred)

submission.to_csv('asmasem_submission.csv', index=False)
submission.head(10)