# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session



import os

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

pd.options.mode.chained_assignment = None

import random

from datetime import datetime

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from matplotlib import pyplot as plt

from scipy import stats



# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# import the relevant Keras modules

from keras.models import Sequential, load_model

from keras.layers import Activation, Dense, LSTM, Dropout, GlobalMaxPooling1D

from keras.optimizers import Adam

from keras.callbacks import ModelCheckpoint, EarlyStopping
DATA_PATH = '/kaggle/input/daily-temperature-of-major-cities/city_temperature.csv'

data = pd.read_csv(DATA_PATH, 

                   dtype={'States': 'string'}, 

                   low_memory=False,

                  )



# Drop columns we won't be using

data.drop(['Region', 'Country', 'State'], inplace=True, axis=1)



# Let's train on one city

input_data = data.loc[data['City'] == 'Chicago']

input_data['Date'] = pd.to_datetime(input_data[['Year', 'Month', 'Day']])

input_data.drop(['Month', 'Day', 'Year'], inplace=True, axis=1)



# Drop outliers

input_data = input_data.loc[(np.abs(stats.zscore(input_data['AvgTemperature'])) < 3)]



input_data.head()
plt.plot(input_data['Date'], input_data['AvgTemperature'])

plt.show()
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):

    n_vars = 1 if type(data) is list else data.shape[1]

    df = pd.DataFrame(data)

    cols, names = list(), list()

    

    # input sequence (t-n, ... t-1)

    for i in range(n_in, 0, -1):

        cols.append(df.shift(i))

        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]

        

    # forecast sequence (t, t+1, ... t+n)

    for i in range(0, n_out):

        cols.append(df.shift(-i))

        if i == 0:

            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]

        else:

            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]

            

    # put it all together

    agg = pd.concat(cols, axis=1)

    agg.columns = names

    

    # drop rows with NaN values

    if dropnan:

        agg.dropna(inplace=True)

        

    return agg
reframed = series_to_supervised(input_data, 1, 1)

reframed.drop(reframed.columns[[0, 2, 3, 5]], inplace=True, axis=1)

reframed.head()
# Train-test split

values = reframed.values

train, test = train_test_split(values)



# Split into input and output

X_train, y_train = train[:, :-1], train[:, -1]

X_test, y_test = test[:, :-1], test[:, -1]



scaler1 = MinMaxScaler()

scaler2 = MinMaxScaler()

X_train = scaler1.fit_transform(X_train)

y_train = scaler2.fit_transform(y_train.reshape(-1, 1))



X_test = scaler1.transform(X_test)

y_test = scaler2.transform(y_test.reshape(-1, 1))





# reshape input to be 3D [samples, timesteps, features]

X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))

X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
callbacks = [

    EarlyStopping(

        monitor='val_loss',

        min_delta=1e-4,

        patience=5,

        verbose=1

    ),

    ModelCheckpoint(

        filepath="weights.h5", 

        monitor="val_loss", 

        mode='min', 

        save_best_only=True,

        save_weights_only=True,

        verbose=1

    )

]



model = Sequential()

model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))

model.add(GlobalMaxPooling1D())

model.add(Dropout(0.25))

model.add(Dense(X_train.shape[2]))



model.compile(loss='mae', optimizer='adam')

model.summary()
model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_test, y_test), 

          verbose=2, shuffle=False, callbacks=callbacks)



model.load_weights('weights.h5')

model.save('model.pb')
model = load_model("model.pb")

plt.plot(scaler2.inverse_transform(y_test.reshape(-1, 1)), label = "actual")

plt.plot(scaler1.inverse_transform(model.predict(X_test)), label = "predicted")

plt.xlabel('Day number')

plt.ylabel('Temperature in F')

plt.legend()

plt.show()