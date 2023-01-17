import os



import math

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns

import warnings



from datetime import datetime
%matplotlib inline

sns.set({'figure.figsize':(10,6), 'axes.titlesize':20, 'axes.labelsize':8})
FILEPATH_TRAIN = os.path.join('../input/train.csv')

FILEPATH_TEST = os.path.join('../input/test.csv')
df_train = pd.read_csv(FILEPATH_TRAIN)

df_train.head()
df_test = pd.read_csv(FILEPATH_TEST)

df_test.head()
df_train.info()
df_train.describe()
plt.hist(df_train[df_train.trip_duration < 5000].trip_duration, bins = 100)

plt.title('Trip duration distribution')

plt.xlabel('Duration of a trip (in seconds)')

plt.ylabel('Number of trips')

plt.show()
df_train = df_train[(df_train.trip_duration < 3000)]

df_train.info()
df_train.isna().sum()
df_train.duplicated().sum()
cat_vars = ['store_and_fwd_flag']
for col in cat_vars:

    df_train[col] = df_train[col].astype('category').cat.codes

df_train.head()
for col in cat_vars:

    df_test[col] = df_test[col].astype('category').cat.codes

df_test.head()
df_train['log_trip_duration'] = np.log(df_train.trip_duration)
df_train['distance'] = np.sqrt((df_train.pickup_latitude - df_train.dropoff_latitude)**2 + (df_train.pickup_longitude - df_train.dropoff_longitude)**2)
df_test['distance'] = np.sqrt((df_test.pickup_latitude - df_test.dropoff_latitude)**2 + (df_test.pickup_longitude - df_test.dropoff_longitude)**2)
df_train['log_distance'] = np.log(df_train.distance)
df_test['log_distance'] = np.log(df_test.distance)
df_train = df_train.drop(['vendor_id', 'store_and_fwd_flag'], axis=1)

df_train.head()
df_test = df_test.drop(['vendor_id', 'store_and_fwd_flag'], axis=1)

df_test.head()
num_features = ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']

target = 'log_trip_duration'
X_train = df_train.loc[:, num_features]

y_train = df_train[target]

X_test = df_test.loc[:, num_features]

X_train.shape, y_train.shape, X_test.shape
from sklearn.ensemble import RandomForestRegressor
m = RandomForestRegressor(n_estimators=20)

m.fit(X_train, y_train)
from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(m,X_train,y_train,cv=5,scoring='neg_mean_squared_log_error')

cv_scores
for i in range(len(cv_scores)):

    cv_scores[i] = np.sqrt(abs(cv_scores[i]))

cv_scores
y_test_pred = m.predict(X_test)

y_test_pred[:5]
submission = pd.DataFrame({'id': df_test.id, 'trip_duration': np.exp(y_test_pred)})

submission.head()
submission.to_csv('Submission_file.csv', index=False)