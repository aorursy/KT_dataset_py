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
df_train =  pd.read_csv('../input/train.csv', nrows = 2_000_000, parse_dates=["pickup_datetime"])

# list first few rows (datapoints)
df_train.head()
print('Old size: %d' % len(df_train))
df_train = df_train.dropna(how = 'any', axis = 'rows')
print('New size: %d' % len(df_train))
df_test = df_train.copy()
#df_my_test['distance'] = np.square(df_my_test['pickup_longitude'] - df_my_test['dropoff_longitude']) + np.square(df_my_test['pickup_latitude'] - df_my_test['dropoff_latitude'])
df_my_test.head()
nyc = (-74.0063889, 40.7141667)
def distance(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295 # Pi/180
    a = 0.5 - np.cos((lat2 - lat1) * p)/2 + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    return 0.6213712 * 12742 * np.arcsin(np.sqrt(a))
df_test['distance_miles'] = distance(df_test.pickup_latitude, df_test.pickup_longitude, \
                                     df_test.dropoff_latitude, df_test.dropoff_longitude)
df_test['distance_to_center'] = distance(nyc[1], nyc[0], \
                                          df_test.dropoff_latitude, df_test.dropoff_longitude)
df_test['hour'] = df_test.pickup_datetime.apply(lambda t: pd.to_datetime(t).hour)
df_test['year'] = df_test.pickup_datetime.apply(lambda t: pd.to_datetime(t).year)
df_test['distance_miles'] > 1000 
X_train = df_test[['distance_miles', 'distance_to_center','passenger_count']]
uh = df_test['hour'] < 8
X_train['costly'] = uh
X_train.head()
uh.head()
idxs = df_my_test['distance'] > 0.1
df_my_test.loc[idxs,'distance'] = 0
idxs = df_my_test['distance'] > 0.1
df_my_test[idxs]
df_my_test['norm_distance'] = df_my_test['distance'] / max(df_my_test['distance'])
df_my_test.head()
max(df_my_test['distance'])
X_train = df_test[['distance_miles','passenger_count']]
X_train.head()

X_train['time'] = df_my_test['']
X_train.head()
y_train = df_my_test['fare_amount']
X_train.head()
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

model_lin = Pipeline((
   #     ("standard_scaler", StandardScaler()),
        ("lin_reg", LinearRegression()),
    ))
model_lin.fit(X_train, y_train)

y_train_pred = model_lin.predict(X_train)

#y_test_pred = model_lin.predict(X_test)

from sklearn.metrics import mean_squared_error
    
rmse = np.sqrt(mean_squared_error(y_train_pred, y_train))
rmse
y_train[1:5]
y_train_pred[1:5]

