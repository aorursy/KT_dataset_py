import numpy as np

import pandas as pd

import math

import sklearn

import sklearn.preprocessing

import datetime

import os

import matplotlib.pyplot as plt

from keras.models import Sequential

from keras.layers import LSTM,Dense

from sklearn.preprocessing import MinMaxScaler

from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
os.listdir("../input")
# import all stock prices 

df = pd.read_csv("../input/time-series/MicrosoftClose.csv").reset_index()

df.head()
# self check data

df.tail(4)
start = min(df.date.tolist())

end = max(df.date.tolist())

print("Start date: ", start, ", End date: ", end)
#Lets convert all column names into lower case

df.columns = [i.lower() for i in df.columns]





#Second lets only select the date and close price columns

try:

    df = df[['date','close']]

except:

    print ("No columns named 'date' or 'close' or both! Please rename your columns.")

    

#Lets rename the columns to seq(sequence) & p(price) for future compatibility

df.columns = ['seq','p']



#Sort seq columns with earliest first

df = df.sort_values(by = ['seq'])
df.head()
# Visualize data

plt.figure(figsize=(40, 4));

plt.subplot(1,2,1);

plt.plot(df.p, color='red', label = 'price')

plt.title('Price')

plt.xlabel('time sequence')

plt.ylabel('price')

plt.legend(loc='best')

plt.show()
min_max_scaler = sklearn.preprocessing.MinMaxScaler()

train_split = int(0.6*len(df))

temp1 , temp2 = df.loc[0:train_split-1], df.loc[train_split:]
#create a scaler function for min-max normalization of stock

min_max_scaler = sklearn.preprocessing.MinMaxScaler()



Scaler = min_max_scaler.fit(temp1[['p']])



temp1['p'] = Scaler.transform(temp1[['p']])

temp2['p'] = Scaler.transform(temp2[['p']])



df = pd.concat([temp1,temp2])
# Visualize data

plt.figure(figsize=(40, 4));

plt.subplot(1,2,1);

plt.plot(df.p, color='red', label = 'price')

plt.title('Scaled Price')

plt.xlabel('time sequence')

plt.ylabel('price')

plt.legend(loc='best')

plt.show()
# function to create train, validation, test data given stock data and sequence length

def load_data(stock, seq_len):

    

    # split data in 60%/20%/20% train/validation/test sets

    valid_set_size_percentage = 20 

    test_set_size_percentage = 20 

    

    data_raw = stock.as_matrix() # convert to numpy array

    data = []

    

    # create all possible sequences of length seq_len

    for index in range(len(data_raw) - seq_len): 

        data.append(data_raw[index: index + seq_len])

    

    data = np.array(data);

    valid_set_size = int(np.round(valid_set_size_percentage/100*data.shape[0]));  

    test_set_size = int(np.round(test_set_size_percentage/100*data.shape[0]));

    train_set_size = data.shape[0] - (valid_set_size + test_set_size);

    

    

    x_train = data[:train_set_size,:-1,:]

    y_train = data[:train_set_size,-1,:]

    

    x_valid = data[train_set_size:train_set_size+valid_set_size,:-1,:]

    y_valid = data[train_set_size:train_set_size+valid_set_size,-1,:]

    

    x_test = data[train_set_size+valid_set_size:,:-1,:]

    y_test = data[train_set_size+valid_set_size:,-1,:]

    

    return [x_train, y_train, x_valid, y_valid, x_test, y_test]
# create train, test data

seq_len = 60 # choose sequence length

x_train, y_train, x_valid, y_valid, x_test, y_test = load_data(df, seq_len)
print('x_train.shape = ',x_train.shape)

print('y_train.shape = ', y_train.shape)

print('x_valid.shape = ',x_valid.shape)

print('y_valid.shape = ', y_valid.shape)

print('x_test.shape = ', x_test.shape)

print('y_test.shape = ',y_test.shape)
plt.figure(figsize=(40, 5));

plt.subplot(1,2,1);

plt.scatter(x_train[0][:,0],x_train[0][:,1], s = 50)

plt.scatter(y_train[0][0],y_train[0][1],s = 50)

plt.xticks(rotation=90)

plt.show()
print ("Y for S1 ",y_train[0])

print ("Last data point X for S2 ",x_train[1][-1,:])
# Choose only open prices

x_train, y_train, x_valid, y_valid, x_test, y_test = x_train[:,:,1], y_train[:,1], x_valid[:,:,1], y_valid[:,1], x_test[:,:,1], y_test[:,1]

print('x_train.shape = ',x_train.shape)

print('y_train.shape = ', y_train.shape)

print('x_valid.shape = ',x_valid.shape)

print('y_valid.shape = ', y_valid.shape)

print('x_test.shape = ', x_test.shape)

print('y_test.shape = ',y_test.shape)
#Build the model

model = Sequential()

model.add(LSTM(256,input_shape=(seq_len-1,1)))

model.add(Dense(1))

model.compile(optimizer='adam',loss='mse')
#Reshape data for (Sample,Timestep,Features) 

x_train = x_train.reshape((x_train.shape[0],x_train.shape[1],1))

x_valid = x_valid.reshape((x_valid.shape[0],x_valid.shape[1],1))

x_test = x_test.reshape((x_test.shape[0],x_test.shape[1],1))
#Fit model with history to check for overfitting

history = model.fit(x_train,y_train,epochs=1,validation_data=(x_valid,y_valid),shuffle=False)
predictions = []

true = []

for i,j in zip(x_test,y_test):

    predictions.append(Scaler.inverse_transform(model.predict(i.reshape(-1,59,1)))[0][0])

    true.append(Scaler.inverse_transform([[j]])[0][0])
plt.figure(figsize=(40, 5));

plt.subplot(1,2,1);

plt.plot(predictions, label = "prediction")

plt.plot(true, label = "actual")

plt.legend(loc='best')

plt.show()
inner_prediction = []

seq = x_test[0]

for i in range(len(x_test)):

    prediction__ = (model.predict(seq.reshape(-1,59,1)))[0][0]

    inner_prediction.append(Scaler.inverse_transform([[prediction__]])[0][0])

    

    #we add append our prediction to the list of sequences and get rid of the first observation

    seq = np.vstack([seq[1:],[[prediction__]]])
plt.figure(figsize=(40, 5));

plt.subplot(1,2,1);

plt.plot(inner_prediction, label = "prediction")

plt.plot(true, label = "actual")

plt.legend(loc='best')

plt.show()