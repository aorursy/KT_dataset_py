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
df=pd.read_csv('/kaggle/input/new-york-city-taxi-fare-prediction/train.csv',parse_dates=['pickup_datetime'], nrows=500000)

df.dropna(inplace=True)
test=pd.read_csv('/kaggle/input/new-york-city-taxi-fare-prediction/test.csv',parse_dates=['pickup_datetime'])

test
df.describe()
test.describe()
# range of longitude for NYC

nyc_min_longitude = -74.3        

nyc_max_longitude = -72

# range of latitude for NYC

nyc_min_latitude = 40.63

nyc_max_latitude = 42

        
# only consider locations within New York City

for long in ['pickup_longitude', 'dropoff_longitude']:

    df = df[(df[long] > nyc_min_longitude) & (df[long] < nyc_max_longitude)]

for lat in ['pickup_latitude', 'dropoff_latitude']:

    df = df[(df[lat] > nyc_min_latitude) & (df[lat] < nyc_max_latitude)]
df.loc[df['passenger_count'] == 0, 'passenger_count'] = 1
df = df[(df['fare_amount'] >0) & (df['fare_amount'] <= 100)]
df.describe()

def euc_distance(lat1, long1, lat2, long2):

        return(((lat1-lat2)**2 + (long1-long2)**2)**0.5)

    

df['travel_distance'] = euc_distance(df['pickup_latitude'], df['pickup_longitude'], df['dropoff_latitude'], df['dropoff_longitude'])

test['travel_distance'] = euc_distance(test['pickup_latitude'], test['pickup_longitude'], test['dropoff_latitude'], test['dropoff_longitude'])
df.corr()
df['year'] = df['pickup_datetime'].dt.year

df['month'] = df['pickup_datetime'].dt.month

df['day'] = df['pickup_datetime'].dt.day

df['day_of_week'] = df['pickup_datetime'].dt.dayofweek

df['hour'] = df['pickup_datetime'].dt.hour

df = df.drop(['pickup_datetime'], axis=1)
df.corr()
test['year'] = test['pickup_datetime'].dt.year

test['month'] =test['pickup_datetime'].dt.month

test['day'] = test['pickup_datetime'].dt.day

test['day_of_week'] =test['pickup_datetime'].dt.dayofweek

test['hour'] = test['pickup_datetime'].dt.hour

test = test.drop(['pickup_datetime'], axis=1)
test.drop(['key'], axis=1,inplace=True)

df.drop(['key'], axis=1,inplace=True)
print(df.isnull().sum())

print(test.isnull().sum())
import pandas as pd

import numpy as np

from sklearn.preprocessing import scale

from sklearn.model_selection import train_test_split

from keras.models import Sequential

from keras.layers import Dense

from sklearn.metrics import mean_squared_error

# Scale the features

df_prescaled = df.copy()

df_scaled = df.drop(['fare_amount'], axis=1)

df_scaled = scale(df_scaled)

cols = df.columns.tolist()

cols.remove('fare_amount')

df_scaled = pd.DataFrame(df_scaled, columns=cols, index=df.index)

df_scaled = pd.concat([df_scaled, df['fare_amount']], axis=1)

df = df_scaled.copy()
# Split the dataframe into a training and testing set

X = df.loc[:, df.columns != 'fare_amount'] 

y = df.fare_amount

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# Build neural network in Keras

model=Sequential()

model.add(Dense(128, activation= 'relu', input_dim=X_train.shape[1]))

model.add(Dense(64, activation= 'relu'))

model.add(Dense(32, activation= 'relu'))

model.add(Dense(8, activation= 'relu'))

model.add(Dense(1))



model.compile(loss='mse', optimizer='adam', metrics=['mse'])



model.fit(X_train, y_train, epochs=16)



# Results

train_pred = model.predict(X_train)

train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))

test_pred = model.predict(X_test)

test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))

print("Train RMSE: {:0.2f}".format(train_rmse))

print("Test RMSE: {:0.2f}".format(test_rmse))

print('------------------------')
test_scaled = scale(test)
pred= model.predict(test_scaled)

pred.shape


submission = pd.read_csv('/kaggle/input/new-york-city-taxi-fare-prediction/sample_submission.csv')

submission['fare_amount'] = pred

submission.to_csv('submission.csv',index=False)