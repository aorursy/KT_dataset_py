# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import os



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns # visualization library

import matplotlib.pyplot as plt # visualization library



import warnings

warnings.filterwarnings('ignore')

%matplotlib inline



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

sample = pd.read_csv('../input/sample_submission.csv')
train.head(10)
train.tail(10)
train.info()
for x in train.keys():

    print(x)
train.isnull().sum()
from datetime import datetime



train['pickup_datetime'] = train['pickup_datetime'].astype('datetime64[ns]')

train['dropoff_datetime'] = train['dropoff_datetime'].astype('datetime64[ns]')
pick_features = ['pickup_datetime', 'dropoff_datetime', 'vendor_id']

pick_df = train[pick_features].copy(True)

pick_df.head()
# Pull out the month, the week,day of week and hour of day and make a new feature for each



pick_df['week'] = pick_df.loc[:,'pickup_datetime'].dt.week;

pick_df['weekday'] = pick_df.loc[:,'pickup_datetime'].dt.weekday;

pick_df['hour'] = pick_df.loc[:,'pickup_datetime'].dt.hour;

pick_df['month'] = pick_df.loc[:,'pickup_datetime'].dt.month;



# Count number of pickups made per month and hour of day

month_usage = pd.value_counts(pick_df['month']).sort_index()

hour_usage = pd.value_counts(pick_df['hour']).sort_index()
figure = plt.subplot(2, 1, 2)

hour_usage.plot.bar(alpha = 0.5, color = 'orange')

plt.title('Pickups over Hour of Day', fontsize = 20)

plt.xlabel('hour', fontsize = 18)

plt.ylabel('Count', fontsize = 18)

plt.xticks(rotation=0)

plt.yticks(fontsize = 18)

plt.show()
figure = plt.subplot(2, 1, 2)

month_usage.plot.bar(alpha = 0.5, color = 'pink')

plt.title('Pickups over Month', fontsize = 20)

plt.xlabel('Month', fontsize = 18)

plt.ylabel('Count', fontsize = 18)

plt.xticks(rotation=0)

plt.yticks(fontsize = 18)

plt.show()
train.passenger_count.min()
train.passenger_count.max()
train.plot.scatter(x='pickup_longitude',y='pickup_latitude')

plt.show()
train.plot.scatter(x='dropoff_longitude',y='dropoff_latitude')

plt.show()
train.trip_duration.min()
train.trip_duration.max()
train.boxplot(figsize=(15,10))

plt.show()
# As said before, there is no need to have the min (0 passenger), we will drop it

train = train[train['passenger_count']>= 1]
# The trip duration's range is between 1 sec to 3526282 sec

# We will drop values that are inferior to 1 min (60 sec) and superior to 166 min (10 000 sec).

train = train[train['trip_duration']>= 1 ]

train = train[train['trip_duration']<= 10000 ]
# We will drop the longitude and latitude (in pickup and dropoff that looks like outliers)

train = train.loc[train['pickup_longitude']> -90]

train = train.loc[train['pickup_latitude']< 47.5]



train = train.loc[train['dropoff_longitude']> -90]

train = train.loc[train['dropoff_latitude']> 34]
col_diff = list(set(train.columns).difference(set(test.columns)))

col_diff
# To use the pickup and dropoff location, we will calculate the distance between them

train['dist'] = abs((train['pickup_latitude']-train['dropoff_latitude'])

                        + (train['pickup_longitude']-train['dropoff_longitude']))

test['dist'] = abs((test['pickup_latitude']-test['dropoff_latitude'])

                        + (test['pickup_longitude']-test['dropoff_longitude']))
y = train["trip_duration"]  # This is our target

X = train[["passenger_count","vendor_id", "pickup_longitude", "pickup_latitude", "dropoff_longitude","dropoff_latitude", "dist" ]]
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import ShuffleSplit

from sklearn.model_selection import cross_val_score
randf = RandomForestRegressor()
randf.fit(X, y)
shuffle = ShuffleSplit(n_splits=5, train_size=0.5, test_size=0.25, random_state=42)
cv_score = cross_val_score(randf, X, y, cv=shuffle, scoring='neg_mean_squared_log_error')

for i in range(len(cv_score)):

    cv_score[i] = np.sqrt(abs(cv_score[i]))

print(np.mean(cv_score))
test.head()
X_test = test[["vendor_id", "passenger_count","pickup_longitude", "pickup_latitude","dropoff_longitude","dropoff_latitude","dist"]]

prediction = randf.predict(X_test)

prediction
my_submission = pd.DataFrame({'id': test.id, 'trip_duration': prediction})

my_submission.head()
my_submission.to_csv('submission.csv', index=False)