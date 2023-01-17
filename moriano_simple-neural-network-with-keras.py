# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
print(os.listdir("../input/bike-sharing-dataset"))

# Any results you write to the current directory are saved as output.
raw = pd.read_csv("../input/bike-sharing-dataset/hour.csv")
raw.head()
raw.describe()
def generate_dummies(df, dummy_column):
    dummies = pd.get_dummies(df[dummy_column], prefix=dummy_column)
    df = pd.concat([df, dummies], axis=1)
    return df

X = pd.DataFrame.copy(raw)
dummy_columns = ["season",     # season (1:springer, 2:summer, 3:fall, 4:winter)
                 "yr",          # year (0: 2011, 1:2012)
                 "mnth",        # month ( 1 to 12)
                 "hr",          # hour (0 to 23)
                 "weekday",     # weekday : day of the week
                 "weathersit"   # weathersit : 
                                 # - 1: Clear, Few clouds, Partly cloudy, Partly cloudy
                                 # - 2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
                                 # - 3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
                                 # - 4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog
                ]
for dummy_column in dummy_columns:
    X = generate_dummies(X, dummy_column)
X.head()
X.columns
for dummy_column in dummy_columns:
    del X[dummy_column]

X.columns
X.head()
X.describe()
first_3_weeks = 3*7*24 # 3 weeks (7 days), 24 hours each day
X[:first_3_weeks].plot(x='dteday', y='cnt', figsize=(18, 5))
del X["instant"]
del X["dteday"]
y = X["cnt"]
del X["cnt"]
del X["registered"]
del X["casual"]
X.head()
all_days = len(X) // 24
print("Total observations", len(X))
print("Total number of days", all_days)
days_for_training = int(all_days * 0.7)
X_train = X[0:days_for_training]
X_test = X[days_for_training:]
print("Observations for training", len(X_train))
print("Observations for testing", len(X_test))
print("Some target values", y.head())
y_normalized = (y - y.min()) / (y.max() - y.min())
y_normalized.head()

y_train = y[0:days_for_training]
y_test = y[days_for_training:]
y_train_normalized = y_normalized[0:days_for_training]
y_test_normalized = y_normalized[days_for_training:]
from keras.models import Sequential
from keras.layers import Dense, Dropout
features = X.shape[1]
model = Sequential()
model.add(Dense(13, input_shape=(features,), activation='relu'))
model.add(Dropout(0.75))
model.add(Dense(1, activation='linear'))

model.summary()
from keras.optimizers import SGD
sgd = SGD(lr=0.01)
model.compile(optimizer=sgd, loss="mean_squared_error")
results = model.fit(X_train, y_train_normalized, epochs=10, validation_data = (X_test, y_test_normalized))
results.history
pd.DataFrame.from_dict(results.history).plot()
