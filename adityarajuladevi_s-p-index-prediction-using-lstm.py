# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from pandas import datetime

from sklearn import preprocessing

from math import sqrt

from keras.layers.recurrent import LSTM

from keras.models import Sequential

from keras.layers import LSTM,Dense, Dropout, Activation

import math, time

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
def normalize_data(df):

    """ Normalize the data in the input dataframe"""

    min_max_scaler = preprocessing.MinMaxScaler()

    df['open'] = min_max_scaler.fit_transform(df.open.values.reshape(-1,1))

    df['high'] = min_max_scaler.fit_transform(df.high.values.reshape(-1,1))

    df['low'] = min_max_scaler.fit_transform(df.low.values.reshape(-1,1))

    df['volume'] = min_max_scaler.fit_transform(df.volume.values.reshape(-1,1))

    df['adj close'] = min_max_scaler.fit_transform(df['adj close'].values.reshape(-1,1))

    return [df,min_max_scaler]
def load_data(stock, seq_len):

    amount_of_features = len(stock.columns) # 5

    data = stock.as_matrix() 

    sequence_length = seq_len + 1 # index starting from 0

    result = []

    

    for index in range(len(data) - sequence_length): # maxmimum date = lastest date - sequence length

        result.append(data[index: index + sequence_length]) # index : index + 22days

    

    result = np.array(result)

    row = round(0.9 * result.shape[0]) # 90% split

    train = result[:int(row), :] # 90% data, all features

    

    x_train = train[:, :-1] 

    y_train = train[:, -1][:,-1]

    

    x_test = result[int(row):, :-1] 

    y_test = result[int(row):, -1][:,-1]

    

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], amount_of_features))

    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], amount_of_features))  



    return [x_train, y_train, x_test, y_test]
def nn_model():

    model = Sequential()

    model.add(Dense(100, input_dim=X_train_NN.shape[1], activation='relu'))

    model.add(Dropout(0.25))

    model.add(Dense(50, activation='relu'))

    model.add(Dropout(0.25))

    model.add(Dense(25, activation='relu'))

    model.add(Dropout(0.25))

    model.add(Dense(1, activation='linear'))

	# Compile model

    model.compile(loss='mse', optimizer='adam')

    return model
def build_model(layers):

    """ Build the LSTM RNN model """

    d = 0.2

    model = Sequential()

    model.add(LSTM(128, input_shape=(layers[1], layers[0]), return_sequences=True))

    model.add(Dropout(d))

    model.add(LSTM(64, input_shape=(layers[1], layers[0]), return_sequences=False))

    model.add(Dropout(d))

    model.add(Dense(16,kernel_initializer='uniform',activation='relu'))  

    model.add(Dense(1,kernel_initializer='uniform',activation='linear'))

    model.compile(loss='mse',optimizer='adam',metrics=['accuracy'])

        

    start = time.time()

    model.compile(loss='mse',optimizer='adam', metrics=['accuracy'])

    print("Compilation Time : ", time.time() - start)

    return model
df = pd.read_csv("../input/SPIndex.csv", index_col = 0)

df["adj close"] = df.adjclose # Moving close to the last column

df.drop(['close','adjclose'], 1, inplace=True) # Moving close to the last column

df.head()
df,min_max_scaler = normalize_data(df)

window=7

X_train, y_train, X_test, y_test = load_data(df, window)



X_train_NN = np.reshape(X_train,(X_train.shape[0],X_train.shape[1]*X_train.shape[2]))

X_test_NN = np.reshape(X_test,(X_test.shape[0],X_test.shape[1]*X_test.shape[2]))
from sklearn import linear_model

logreg = linear_model.LinearRegression()

res = logreg.fit(X_train_NN,y_train)

y_lr=res.predict(X_test_NN)
#Fitting to the ANN'

classifier = nn_model()

classifier.fit(X_train_NN,y_train,epochs=100, batch_size=10)

y_pred=classifier.predict(X_test_NN)
model = build_model([5,window,1])

history = model.fit(X_train,y_train,batch_size=512,epochs=150,validation_split=0.1,verbose=1)

y_lstm = model.predict(X_test)
plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.xlabel('epochs')

plt.ylabel('loss')

plt.legend(['train','validation'])
import matplotlib.pyplot as plt2

plt2.plot(y_pred,color='red', label='y_mlp')

plt2.plot(y_test,color='blue', label='y_test')

plt2.plot(y_lstm,color='green',label='y_lstm')

plt2.legend(loc='upper left')

plt2.show()