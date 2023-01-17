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
data = pd.read_csv('/kaggle/input/population-time-series-data/POP.csv')



data.head()
data = data[['date', 'value']]

data.head()
data.date.tail()
import matplotlib.pyplot as plt
plt.plot(data.date, data.value)
data.date = pd.to_datetime(data.date)
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

plt.plot(data.date, data.value)
len(data)
data.describe()
data.info()
sigma = data.value.std()

nu = data.value.mean()

def z(x):

    return (x-nu)/sigma

data.value = data.value.apply(z)
import seaborn as sns
sns.boxplot(data.value)
train_data = data[:720]

test_data = data[720:]
# creating training data

lookback = 11  # creating timestep of 12 months i.e 1trainyear



trainx = []

trainy = []

temp = []

for i in range(len(train_data.value)-lookback):

    temp.append(train_data.value[i:i+lookback])

    trainy.append(train_data.value[i+lookback])

    trainx.append(temp)

    temp = []
test_data.head()
# creating testing data

lookback = 11 

testx = []

testy = []

temp = []

for i in range(len(test_data.value)-lookback):

    temp.append(test_data.value.iloc[i:i+lookback])

    testy.append(test_data.value.iloc[i+lookback])

    testx.append(temp)

    temp = []

tdata=test_data.iloc[:i+1]
trainx = np.array(trainx)

trainy = np.array(trainy)

testx = np.array(testx)

testy = np.array(testy)
trainx = trainx.reshape(trainx.shape[0], 11,1)

testx = testx.reshape(testx.shape[0],11, 1)
import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import *
model = Sequential([

    Bidirectional(LSTM(32, activation='relu', input_shape=(11, 1), return_sequences=True)),

#     Dropout(0.5),

    Bidirectional(LSTM(64, activation='relu',  return_sequences=True)),

#     Bidirectional(LSTM(128, activation='relu', return_sequences=True)),

#     Dropout(0.5),

    Bidirectional(LSTM(256, activation='elu')),

#     Dropout(0.5),

    Dense(1028, activation='relu'),

    Dense(1)

])
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.summary()
history = model.fit(trainx, trainy, batch_size=64, validation_split=0.2, epochs=100, )
plt.plot(history.history['val_loss'])

plt.plot(history.history['loss'])

plt.legend(['val_loss', 'loss'])
plt.plot(history.history['val_mse'])

plt.plot(history.history['mse'])

plt.legend(['val_mse', 'mse'])
pred = model.predict(testx)



tdata=test_data.iloc[:i+1]



# pred = pred.flatten()



tdata['pred'] = pred
plt.figure(figsize=[10, 5])

plt.plot(train_data.value)

plt.plot(test_data.value, )

plt.plot(tdata.pred)