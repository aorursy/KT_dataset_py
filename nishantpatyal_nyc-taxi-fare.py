# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



sample_sub = pd.read_csv('/kaggle/input/new-york-city-taxi-fare-prediction/sample_submission.csv')

train_df = pd.read_csv('/kaggle/input/new-york-city-taxi-fare-prediction/train.csv', nrows = 500000)

test_df = pd.read_csv('/kaggle/input/new-york-city-taxi-fare-prediction/test.csv')



test_df['pickup_datetime'] = pd.to_datetime(test_df['pickup_datetime'])

train_df['pickup_datetime'] = pd.to_datetime(train_df['pickup_datetime'])



key = test_df['key']

test_df = test_df.drop('key',1)[['pickup_datetime',

                                   'pickup_longitude',

                                   'pickup_latitude',

                                   'dropoff_longitude',

                                   'dropoff_latitude',

                                   'passenger_count',]]



train_df = train_df.drop('key',1)[['pickup_datetime',

                                   'pickup_longitude',

                                   'pickup_latitude',

                                   'dropoff_longitude',

                                   'dropoff_latitude',

                                   'passenger_count',

                                   'fare_amount']]
train_df = train_df[train_df['fare_amount']>0]

train_df = train_df[train_df['fare_amount']<100]
def distance(lat1, lon1, lat2, lon2):

    p = 0.017453292519943295 # Pi/180

    a = 0.5 - np.cos((lat2 - lat1) * p)/2 + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2

    return 0.6213712 * 12742 * np.arcsin(np.sqrt(a))



train_df['distance_miles'] = distance(train_df.pickup_latitude, train_df.pickup_longitude, \

                                      train_df.dropoff_latitude, train_df.dropoff_longitude)



test_df['distance_miles'] = distance(test_df.pickup_latitude, test_df.pickup_longitude, \

                                      test_df.dropoff_latitude, test_df.dropoff_longitude)
train_df = train_df.drop(['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude'],1)

train_df['hour_of_day'] = train_df['pickup_datetime'].dt.hour

train_df['dayofyear'] = train_df['pickup_datetime'].dt.dayofyear

train_df['month'] = train_df['pickup_datetime'].dt.month

train_df['year'] = train_df['pickup_datetime'].dt.year

train_df['year'] = train_df['year'].apply(lambda X: int(str(X)[-2:]))

train_df = train_df.drop('pickup_datetime',1)

train_df = train_df[train_df['distance_miles']<17]





test_df = test_df.drop(['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude'],1)

test_df['hour_of_day'] = test_df['pickup_datetime'].dt.hour

test_df['dayofyear'] = test_df['pickup_datetime'].dt.dayofyear

test_df['month'] = test_df['pickup_datetime'].dt.month

test_df['year'] = test_df['pickup_datetime'].dt.year

test_df['year'] = test_df['year'].apply(lambda X: int(str(X)[-2:]))

test_df = test_df.drop('pickup_datetime',1)
X = train_df.drop('fare_amount',1).values

y = train_df['fare_amount'].values



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)



from sklearn.preprocessing import MinMaxScaler



sclr = MinMaxScaler()



X_train = sclr.fit_transform(X_train)

X_test = sclr.transform(X_test)



#for final predicion

test_df = sclr.transform(test_df)
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout

from tensorflow.keras.callbacks import EarlyStopping



model = Sequential()

model.add(Dense(6, activation = 'relu'))

model.add(Dense(15, activation = 'relu'))

model.add(Dense(7, activation = 'relu'))

model.add(Dense(3, activation = 'relu'))



model.add(Dense(1))



model.compile(optimizer = 'adam',

             loss = 'mse')







es = EarlyStopping(monitor='val_loss',

                  mode = 'min',

                  verbose=1,

                  patience=2)



model.fit(X_train,y_train,

          epochs=500,

          verbose=1,

          validation_data=(X_test, y_test),

          callbacks=[es])
model.evaluate(X_test,y_test)
sub = model.predict(test_df)



s = pd.DataFrame(key)

t = pd.DataFrame(sub)



s.join(t).rename(columns = {0:'fare_amount'}).to_csv('Submission.csv', index = None)