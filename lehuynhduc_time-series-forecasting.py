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
data = pd.read_csv('/kaggle/input/AirPassengers.csv')
data['Month'] = pd.to_datetime(data['Month'])

data.set_index('Month',inplace=True)
train_df = data.loc[:'1958-12-01']

test_df = data.loc['1959-01-01':]
train_df.tail()
test_df.head()
import matplotlib.pyplot as plt

%matplotlib inline
plt.plot(train_df)
def split_sequence(sequence, n_steps):

    X, y = list(), list()

    for i in range(len(sequence)):

        end_ix = i + n_steps

        if end_ix > len(sequence)-1:

            break

        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]

        X.append(seq_x)

        y.append(seq_y)

    

    return np.array(X), np.array(y)



def mean_absolute_percentage_error(a, b): 

    a = np.array(a)

    b = np.array(b)

    mask = a != 0

    return (np.abs(a - b)/a)[mask].mean()*100



def smape(a, b): 

    a = np.array(a)

    b = np.array(b)

    mask = a != 0

    return (np.abs(a - b)/(np.abs(a)+np.abs(b)))[mask].mean()*100
from keras.models import Sequential

from keras.layers import LSTM

from keras.layers import Dense
t = train_df['#Passengers'].values

n_steps = 12

X, y = split_sequence(t, n_steps)
def predict_lstm(t,X,y,n_steps):

    # reshape from [samples, timesteps] into [samples, timesteps, features]

    n_features = 1

    X = X.reshape((X.shape[0], X.shape[1], n_features))

    # define model

    model = Sequential()

    model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))

    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mae')

    model.fit(X, y, epochs=200, verbose=1)

    

    seq = t[-n_steps:]

    pred = []

    for i in range(24):

        x_input = seq[-n_steps:]

        x_input = x_input.reshape((1, n_steps, n_features))

        yhat = model.predict(x_input, verbose=0)

        seq = np.append(seq,yhat[0])

        pred.append(yhat[0][0])

    return pred
pred = predict_lstm(t,X,y,n_steps)
print("MAPE: ", mean_absolute_percentage_error(test_df["#Passengers"].values,pred))

print("SMAPE: ", smape(test_df["#Passengers"].values,pred))
test_df['pred'] = pred

plt.figure(figsize=(17,10))

plt.plot(test_df['#Passengers'],label='#Passengers')

plt.plot(test_df['pred'], label='predict')

plt.legend(loc='lower center')
from sklearn.linear_model import LinearRegression, Lasso, Ridge
def predict_linear(t,X,y,n_steps):

    model = Ridge()

    model.fit(X, y)

    seq = t[-n_steps:]

    pred = []

    for i in range(24):

        x_input = seq[-n_steps:]

        yhat = model.predict([x_input])

        seq = np.append(seq,yhat[0])

        pred.append(yhat[0])

    return pred
pred = predict_linear(t,X,y,n_steps)
print("MAPE: ", mean_absolute_percentage_error(test_df["#Passengers"].values,pred))

print("SMAPE: ", smape(test_df["#Passengers"].values,pred))
test_df['pred'] = pred

plt.figure(figsize=(17,10))

plt.plot(test_df['#Passengers'],label='#Passengers')

plt.plot(test_df['pred'], label='predict')

plt.legend(loc='lower center')