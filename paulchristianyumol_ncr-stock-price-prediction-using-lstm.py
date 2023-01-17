import math

import pandas_datareader as web

import numpy as np

from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential

from keras.layers import Dense, LSTM, Dropout

from keras.utils import plot_model

import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')
df = web.DataReader('NCR', data_source='yahoo', start='2015-01-01', end='2020-7-15')  

df
df.shape
plt.figure(figsize=(16,8))

plt.title('Adjusted Close Price History')

plt.plot(df['Adj Close'])

plt.xlabel('Date',fontsize=18)

plt.ylabel('Adjusted Close Price USD ($)',fontsize=18)

plt.show()
data = df.filter(['Adj Close'])

dataset = data.values

training_data_len = math.ceil( len(dataset) *.8)

training_data_len
scaler = MinMaxScaler(feature_range=(0, 1)) 

scaled_data = scaler.fit_transform(dataset)
train_data = scaled_data[0:training_data_len  , : ]

x_train = []

y_train = []

for i in range(60,len(train_data)):

    x_train.append(train_data[i-60:i,0])

    y_train.append(train_data[i,0])
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

x_train.shape
test_data = scaled_data[training_data_len - 60: , : ]

x_test = []

y_test = dataset[training_data_len : , : ] 

for i in range(60,len(test_data)):

    x_test.append(test_data[i-60:i,0])
x_test= np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))

x_test.shape
model = Sequential()

model.add(LSTM(units=256, return_sequences=True, input_shape=(x_train.shape[1],1)))

model.add(Dropout(0.3))

model.add(LSTM(units=256, return_sequences=False))

model.add(Dropout(0.3))

model.add(Dense(units=32))

model.add(Dense(units=1))



model.summary()
plot_model(model)
model.compile('adam', 'mse')
hist = model.fit(x_train, y_train, batch_size=1, epochs=3)
predictions = model.predict(x_test) 

predictions = scaler.inverse_transform(predictions) #Undo scaling
rmse=np.sqrt(np.mean(((predictions - y_test)**2)))

rmse
train = data[:training_data_len]

valid = data[training_data_len:]

valid['Predictions'] = predictions



plt.figure(figsize=(16,8))

plt.title('Valid VS Prediction')

plt.xlabel('Date', fontsize=18)

plt.ylabel('Adjusted Close Price USD ($)', fontsize=18)

plt.plot(valid['Adj Close'], color='C1')

plt.plot(valid['Predictions'], color='C2')

plt.legend(['Valid', 'Predictions'], loc='lower right')

plt.show()



plt.figure(figsize=(16,8))

plt.title('Model of the Overall Data')

plt.xlabel('Date', fontsize=18)

plt.ylabel('Adjusted Close Price USD ($)', fontsize=18)

plt.plot(train['Adj Close'])

plt.plot(valid[['Adj Close', 'Predictions']])

plt.legend(['Train', 'Valid', 'Predictions'], loc='lower right')

plt.show()
valid
#Get the quote

NCR_quote = web.DataReader('NCR', data_source='yahoo', start='2015-01-01', end='2020-7-15')



#Create a new dataframe

new_df = NCR_quote.filter(['Adj Close'])



#Get the last 60 day closing price 

last_60_days = new_df[-60:].values



#Scale the data to be values between 0 and 1

last_60_days_scaled = scaler.transform(last_60_days)



#Create an empty list

X_test = []



#Append the past 60 days

X_test.append(last_60_days_scaled)



#Convert the X_test data set to a numpy array

X_test = np.array(X_test)



#Reshape the data

X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))



#Get the predicted scaled price

pred_price = model.predict(X_test)

                           

#undo the scaling 

pred_price = scaler.inverse_transform(pred_price)

print(pred_price)
#Get the quote

NCR_quote2 = web.DataReader('NCR', data_source='yahoo', start='2020-7-16', end='2020-7-16')

print(NCR_quote2['Adj Close'])