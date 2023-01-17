# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import tensorflow as tf

import matplotlib.pyplot as plt

import sklearn.preprocessing

from sklearn.metrics import r2_score



from keras.layers import Dense,Dropout,LSTM

from keras.models import Sequential



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# read data from AEP hourly

fpath='/kaggle/input/hourly-energy-consumption/AEP_hourly.csv'

df=pd.read_csv(fpath)

df.head()
# change index column from numbers 0, 1, 2.. to date&time information

df = pd.read_csv(fpath, index_col='Datetime', parse_dates=['Datetime'])

df.head()
# check missing data in file

df.isna().sum()
# prepare data for training

def load_data(stock, seq_len):

    X_train = []

    y_train = []

    for i in range(seq_len, len(stock)):

        X_train.append(stock.iloc[i-seq_len : i, 0])

        y_train.append(stock.iloc[i, 0])

    

    #last 1000 days are going to be used in test

    X_test = X_train[97272:]             

    y_test = y_train[97272:]

    

    #first 4053 days are going to be used in training

    X_train = X_train[:97272]           

    y_train = y_train[:97272]

    

    # convert to numpy array

    X_train = np.array(X_train)

    y_train = np.array(y_train)

    X_test = np.array(X_test)

    y_test = np.array(y_test)

    

    #4 reshape data to input into RNN models

    X_train = np.reshape(X_train, (97272, seq_len, 1))

    X_test = np.reshape(X_test, (X_test.shape[0], seq_len, 1))

    

    return [X_train, y_train, X_test, y_test]



#create train, test data

seq_len = 24 #choose sequence length



X_train, y_train, X_test, y_test = load_data(df, seq_len)



print('X_train.shape = ',X_train.shape)

print('y_train.shape = ', y_train.shape)

print('X_test.shape = ', X_test.shape)

print('y_test.shape = ',y_test.shape)
# build LSTM model

lstm_model = Sequential()



lstm_model.add(LSTM(40,activation="tanh",return_sequences=True, input_shape=(X_train.shape[1],1)))

lstm_model.add(Dropout(0.15))



lstm_model.add(LSTM(40,activation="tanh",return_sequences=True))

lstm_model.add(Dropout(0.15))



lstm_model.add(LSTM(40,activation="tanh",return_sequences=False))

lstm_model.add(Dropout(0.15))



lstm_model.add(Dense(1))



lstm_model.summary()



# train LSTM model

lstm_model.compile(optimizer="adam",loss="MSE")

lstm_model.fit(X_train, y_train, epochs=10, batch_size=1000)
lstm_predictions = lstm_model.predict(X_test)



lstm_score = r2_score(y_test, lstm_predictions)

print("R^2 Score of LSTM model = ",lstm_score)
def plot_predictions(test, predicted, title):

    plt.figure(figsize=(16,4))

    plt.plot(test, color='blue',label='Actual power consumption data')

    plt.plot(predicted, alpha=0.7, color='orange',label='Predicted power consumption data')

    plt.title(title)

    plt.xlabel('Time')

    plt.ylabel('Normalized power consumption scale')

    plt.legend()

    plt.show()
plot_predictions(y_test, lstm_predictions, "Predictions made by LSTM model")