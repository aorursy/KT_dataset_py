import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')



import os

print(os.listdir("../input"))





sns.set_style('whitegrid')



params={'legend.fontsize':'x-large',

       'figure.figsize':(20,10),

       'axes.labelsize':'x-large',

       'axes.titlesize':'x-large',

       'xtick.labelsize':'x-large',

       'ytick.labelsize':'x-large'}

plt.rcParams.update(params)
gs_df_train=pd.read_csv('../input/Google_Stock_Price_Train.csv')

print(gs_df_train.shape)

gs_df_train.head()
training_set=gs_df_train.iloc[:,1:2].values

from sklearn.preprocessing import MinMaxScaler

sc=MinMaxScaler(feature_range=(0,1))

training_sc_set=sc.fit_transform(training_set)

training_sc_set
X_train=[]

y_train=[]

for i in range(10,1258):

    X_train.append(training_sc_set[i-10:i,0])

    y_train.append(training_sc_set[i,0])

X_train,y_train=np.array(X_train),np.array(y_train)

print(X_train.shape,y_train.shape)

X_train=np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))

X_train.shape
import keras

from keras.models import Sequential

from keras.layers import LSTM

from keras.layers import Dense

from keras.layers import Dropout,Flatten

from keras import backend as k
model=Sequential()

model
model.add(LSTM(units=50,return_sequences=True,input_shape=(10,1)))

model.add(Dropout(0.2))

#model.add(LSTM(units=50,return_sequences=True))

#model.add(Dropout(0.2))

model.add(Flatten(data_format=None))

model.add(Dense(units=1))

model.compile(optimizer='RMSprop',loss='mean_squared_error')



model.fit(X_train,y_train,epochs=50,batch_size=32)
gs_df_test=pd.read_csv('../input/Google_Stock_Price_Test.csv')

print(gs_df_test.shape)

gs_df_test.head()
real_gs_price=gs_df_test.iloc[:,1:2].values

real_gs_price
gs_df_total=pd.concat([gs_df_train['Open'],gs_df_test['Open']],axis=0)

print(gs_df_total.shape)

gs_df_total.head()
test_input=gs_df_total[len(gs_df_total)-len(gs_df_test)-10:].values

test_input
test_input=test_input.reshape(-1,1)

print(test_input.shape)

test_input
test_input=sc.transform(test_input)

print(test_input.shape)

test_input
X_test=[]

for i in range(10,31):

    X_test.append(test_input[i-10:i,0])



X_test=np.array(X_test)

print(X_test.shape)

X_test
X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))

print(X_test.shape)

X_test
predicted_gs_price=model.predict(X_test)

predicted_gs_price
predicted_gs_price=sc.inverse_transform(predicted_gs_price)

predicted_gs_price
plt.plot(real_gs_price,color='r',alpha=0.3,label='Real stock price')

plt.plot(predicted_gs_price,color='g',label='Predicted stock price')

plt.title('Google Stock Price Prediction')

plt.xlabel('Time')

plt.ylabel('Google Stock Price')

plt.legend()

plt.show()
gs_training_set=gs_df_train.iloc[:,4:5].values

gs_training_set
scaler=MinMaxScaler(feature_range=(0,1))

train_input=scaler.fit_transform(gs_training_set)

print(train_input.shape)

train_input
gs_X_train=[]

gs_y_train=[]

for i in range(10,1258):

    gs_X_train.append(train_input[i-10:i,0])

    gs_y_train.append(train_input[i,0])

gs_X_train,gs_y_train=np.array(gs_X_train),np.array(gs_y_train)

print(gs_X_train.shape,gs_y_train.shape)
gs_X_train=np.reshape(gs_X_train,(gs_X_train.shape[0],gs_X_train.shape[1],1))

print(gs_X_train.shape)

gs_X_train
regressor=Sequential()

regressor
regressor.add(LSTM(units=100,return_sequences=True,input_shape=(10,1)))

regressor.add(Dropout(0.2))

#regressor.add(LSTM(units=100,return_sequences=True))

#regressor.add(Dropout(0.2))

regressor.add(Flatten(data_format=None))

regressor.add(Dense(units=1))

regressor.compile(optimizer="RMSprop",loss='mean_squared_error')



regressor.fit(gs_X_train,gs_y_train,epochs=50,batch_size=32)
real_gstock_price=gs_df_test.iloc[:,4:5].values

print(real_gstock_price.shape)

real_gstock_price
input_total=pd.concat([gs_df_train['Close'],gs_df_test['Close']],axis=0)

input_total.head()
input_test=input_total[len(input_total)-len(gs_df_test)-10:].values

print(input_test.shape)

input_test
input_test=input_test.reshape(-1,1)

input_test
input_test=scaler.transform(input_test)

print(input_test.shape)

input_test
gs_X_test=[]

for i in range(10,31):

    gs_X_test.append(input_test[i-10:i,0])

gs_X_test=np.array(gs_X_test)

print(gs_X_test.shape)

gs_X_test
gs_X_test=np.reshape(gs_X_test,(gs_X_test.shape[0],gs_X_test.shape[1],1))

print(gs_X_test.shape)

gs_X_test
predicted_gstock_price=regressor.predict(gs_X_test)

predicted_gstock_price
predicted_gstock_price=scaler.inverse_transform(predicted_gstock_price)

predicted_gstock_price
plt.plot(real_gstock_price,color='r',alpha=0.3,label='Real stock price')

plt.plot(predicted_gstock_price,color='g',label='Predicted stock price')

plt.title('Google Stock Price Prediction')

plt.xlabel('Time')

plt.ylabel('Google Stock Price')

plt.legend()

plt.show()