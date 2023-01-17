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
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn import preprocessing

from keras.layers import Dense,Dropout,LSTM
from keras.models import Sequential
from keras.losses import mean_squared_error
from keras.backend import sign
from keras.utils import to_categorical
from sklearn.utils import shuffle
fpath='/kaggle/input/eye_state.csv'
df=pd.read_csv(fpath, header=0, names=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12',
       '13', '14', 'Target'])
df.head()
# check missing data in file
df.isna().sum()
df.max()
df.mean()
df.shape
def normalize_data(df):
    for each in df:
        scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(0, 1)) #define the range according to each column !!!
        df[each]=scaler.fit_transform(df[each].values.reshape(-1,1))
    return df

from sklearn.model_selection import train_test_split

target = df['Target']
target = to_categorical(target)
df = df.drop(columns=['Target'])

df_norm = normalize_data(df)
df_norm = np.array(df_norm)

X_train, X_test, y_train, y_test = train_test_split(df_norm, target, test_size=0.3, shuffle=True)
df_norm.shape
print("Train data shape ", X_train.shape," ", y_train.shape, "test data shape ",X_test.shape," ", y_test.shape)
X_train = np.reshape(X_train, (10485, 14, 1))
X_test = np.reshape(X_test, (4494, 14, 1))
df_norm[:5]
model = Sequential()

model.add(LSTM(128, input_shape=(X_train.shape[1], 1), return_sequences=True))
model.add(LSTM(32))
model.add(Dropout(0.25))
model.add(Dense(2, activation='softmax'))

model.summary()

model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])
model.fit(X_train, y_train, epochs=10, validation_split=0.2)
model.evaluate(X_test, y_test)
prediction = model.predict(X_test)
prediction[:5], y_test[:5]
