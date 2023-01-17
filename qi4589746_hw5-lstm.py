import numpy as np

import pandas as pd

import matplotlib.pyplot as plt2

import math, time

from sklearn import preprocessing

from sklearn.metrics import mean_squared_error

from math import sqrt

from keras.models import Sequential

from keras.layers.core import Dense, Dropout, Activation

from keras.optimizers import Adam

from keras.layers import LSTM , GRU

import keras

import os
df = pd.read_csv("../input/prices-split-adjusted.csv", index_col = 0)

df["adj close"] = df.close

df.drop(['close'], 1, inplace=True)

df.head()
df = df[df.symbol == 'LLL']

df.drop(['symbol'],1,inplace=True)

df.head()
def normalize_data(df):

    min_max_scaler = preprocessing.MinMaxScaler()

    df['open'] = min_max_scaler.fit_transform(df.open.values.reshape(-1,1))

    df['high'] = min_max_scaler.fit_transform(df.high.values.reshape(-1,1))

    df['low'] = min_max_scaler.fit_transform(df.low.values.reshape(-1,1))

    df['volume'] = min_max_scaler.fit_transform(df.volume.values.reshape(-1,1))

    df['adj close'] = min_max_scaler.fit_transform(df['adj close'].values.reshape(-1,1))

    return df
df = normalize_data(df)

df.head()
def generate_training_data(stock, seq_len):

    amount_of_features = len(stock.columns)

    data = stock.as_matrix() 

    sequence_length = seq_len + 1

    result = []

    

    for index in range(len(data) - sequence_length):

        result.append(data[index: index + sequence_length])

    

    result = np.array(result)

    row = round(0.9 * result.shape[0])

    train = result[:int(row), :]

    

    x_train = train[:, :-1] 

    y_train = train[:, -1][:,-1]

    

    x_test = result[int(row):, :-1] 

    y_test = result[int(row):, -1][:,-1]



    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], amount_of_features))

    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], amount_of_features))  



    return [x_train, y_train, x_test, y_test]
def build_model2(layers):

    d = 0.3

    model = Sequential()

    model.add(GRU(256 , input_shape = (layers[1], layers[0]) , return_sequences=True))

    model.add(Dropout(d))

    model.add(LSTM(256))

    model.add(Dropout(d))

    model.add(Dense(64 ,  activation = 'relu'))

    model.add(Dense(1))

    print(model.summary())

    start = time.time()

    model.compile(loss='mean_squared_error', optimizer=Adam(lr = 0.0005) , metrics = ['mean_squared_error'])

    print("Compilation Time : ", time.time() - start)

    return model
window = 60

X_train, y_train, X_test, y_test = generate_training_data(df, window)
model = build_model2([5,window,1])
model.fit(X_train,y_train,batch_size=512,epochs=90,validation_split=0.1,verbose=1)
diff=[]

ratio=[]

p = model.predict(X_test)

print (p.shape)

for u in range(len(y_test)):

    pr = p[u][0]

    ratio.append((y_test[u]/pr)-1)

    diff.append(abs(y_test[u]- pr))
df = pd.read_csv("../input/prices-split-adjusted.csv", index_col = 0)

df["adj close"] = df.close

df.drop(['close'], 1, inplace=True)

df = df[df.symbol == 'LLL']

df.drop(['symbol'],1,inplace=True)



def denormalize(df, normalized_value): 

    df = df['adj close'].values.reshape(-1,1)

    normalized_value = normalized_value.reshape(-1,1)

    

    min_max_scaler = preprocessing.MinMaxScaler()

    a = min_max_scaler.fit_transform(df)

    new = min_max_scaler.inverse_transform(normalized_value)

    return new



newp = denormalize(df, p)

newy_test = denormalize(df, y_test)
def model_score(model, X_train, y_train, X_test, y_test):

    trainScore = model.evaluate(X_train, y_train, verbose=0)

    print('Train Score: %.5f MSE (%.2f RMSE)' % (trainScore[0], math.sqrt(trainScore[0])))



    testScore = model.evaluate(X_test, y_test, verbose=0)

    print('Test Score: %.5f MSE (%.2f RMSE)' % (testScore[0], math.sqrt(testScore[0])))

    return trainScore[0], testScore[0]





model_score(model, X_train, y_train, X_test, y_test)
import matplotlib.pyplot as plt2



plt2.plot(newp,color='red', label='Prediction')

plt2.plot(newy_test,color='blue', label='Actual')

plt2.legend(loc='best')

plt2.show()
def estimation_error(actual, prediction):

    return (actual - prediction)**2



my_estimation_error = estimation_error(newy_test, newp)

for a in my_estimation_error:

    print (a[0])
plt2.plot(newp,color='red', label='Prediction')

plt2.plot(newy_test,color='blue', label='Actual')

plt2.plot(my_estimation_error,color='orange', label='Estimation_error')

plt2.legend(loc='best')

plt2.show()