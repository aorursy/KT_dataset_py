#import packages

import pandas as pd

import numpy as np



import matplotlib.pyplot as plt

%matplotlib inline



#setting figure size

from matplotlib.pylab import rcParams

rcParams['figure.figsize'] = 16,5



from matplotlib.lines import Line2D



#for normalizing data

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))



from keras.models import Sequential

from keras.layers import Dense, Dropout, LSTM



from datetime import datetime



vTicker = 'KEGN'





#read datafile

#df = pd.read_csv('data/NSE_KPLC_KenyaPower.csv')

df = pd.read_csv('data/NSE_KEGN_Kengen.csv')



#print the head

print(vTicker)

df.head()
#Format date to yyyymmdd and index data by date



df['Date'] = pd.to_datetime(df.Date,format='%d-%m-%Y')

df.index = df['Date']



#plot

plt.plot(df['Close'])

plt.title('[ ' + vTicker + ' ] - CLOSING PRICE TREND')

plt.show()
#Create new dataFrame with Date & Close price



data = df.sort_index(ascending=True, axis=0)

new_data = pd.DataFrame(index=range(0,len(df)),columns=['Date', 'Close'])



for i in range(0,len(data)):

     new_data['Date'][i] = data['Date'][i]

     new_data['Close'][i] = data['Close'][i]



new_data.head()
#Splitting training & validation datasets



#set date index

new_data.index = new_data.Date

new_data.drop('Date', axis=1, inplace=True)



#creating train and test sets

dataset = new_data.values



#Split data for training - 80% and Validation 20%

vCount = round(len(dataset)*0.8)

#print(vCount)



train = dataset[0:vCount,:]

valid = dataset[vCount:,:]



dataset.shape, train.shape, valid.shape



#for i in range(5):

#    print(train[i])



#converting dataset into x_train and y_train

scaler = MinMaxScaler(feature_range=(0, 1))

scaled_data = scaler.fit_transform(dataset)



# Use previous 60 days data to predict

x_train, y_train = [], []

for i in range(60,len(train)):

    x_train.append(scaled_data[i-60:i,0])

    y_train.append(scaled_data[i,0])

    

x_train, y_train = np.array(x_train), np.array(y_train)



x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))



# create and fit the LSTM network

model = Sequential()

model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))

model.add(LSTM(units=50))

model.add(Dense(1))



model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(x_train, y_train, epochs=2, batch_size=2, verbose=2)



#predicting 246 values, using past 60 from the train data

inputs = new_data[len(new_data) - len(valid) - 60:].values

inputs = inputs.reshape(-1,1)

inputs  = scaler.transform(inputs)



X_test = []

for i in range(60,inputs.shape[0]):

    X_test.append(inputs[i-60:i,0])

X_test = np.array(X_test)



X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))

closing_price = model.predict(X_test)

closing_price = scaler.inverse_transform(closing_price)
rms=np.sqrt(np.mean(np.power((valid-closing_price),2)))

rms
#for plotting

train = new_data[:vCount]

valid = new_data[vCount:]



valid['Predictions'] = closing_price

plt.plot(train['Close'], label='Closing Price Trend', lw=2)



#plt.plot(valid[['Close','Predictions']], dashes=[3, 2], label='Actual Vs. Predicted Closing Price', lw=2)

plt.plot(valid['Close'], label='Actual Closing Price', lw=1)

plt.plot(valid['Predictions'], dashes=[1, 2], label='Predicted Price', lw=2)



plt.title('[ ' + vTicker + ' ] - ACTUAL Vs. PREDICTED CLOSING PRICE').set_color('green')

plt.ylabel('Share Price (KES.)')

plt.xlabel('Trading Date')



plt.legend()

plt.show()
