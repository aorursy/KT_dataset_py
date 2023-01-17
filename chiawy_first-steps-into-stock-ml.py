# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

%matplotlib inline



#setting figure size

from matplotlib.pylab import rcParams

rcParams['figure.figsize'] = 16,5



from matplotlib.lines import Line2D



#for normalizing data

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

scaler = MinMaxScaler(feature_range=(0, 1))



from keras.models import Sequential

from keras.layers import Dense, Dropout, LSTM



from datetime import datetime



vTicker = 'KEGN'





#read datafile

#df = pd.read_csv('data/'NSE_SCOM_Safaricom.csv')

df = pd.read_csv('../input/NSE_SCOM_Safaricom.csv')



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

new_data = df.copy()

new_data = new_data.drop(['Open', 'High', 'Low', 'Vol.', 'Change %', 'Date'], axis=1)

new_data.head()
#Split the data into 80% train, 20% test

vCount = round(new_data.shape[0]*0.8)



dataset = new_data.values



train = dataset[0:vCount,:]

valid = dataset[vCount:,:]



new_data.shape, train.shape, valid.shape
scaler = MinMaxScaler(feature_range = (0, 1))

training_data = scaler.fit_transform(train)



features_set = []

labels = []



for i in range(60, train.shape[0]):

    features_set.append(training_data[i-60:i, 0])

    labels.append(training_data[i, 0])



features_set, labels = np.array(features_set), np.array(labels)

features_set = np.reshape(features_set, (features_set.shape[0], features_set.shape[1], 1))



model = Sequential()

model.add(LSTM(units=50, return_sequences=True, input_shape=(features_set.shape[1], 1)))

model.add(Dropout(0.2))



model.add(LSTM(units=50, return_sequences=True))

model.add(Dropout(0.2))



model.add(LSTM(units=50, return_sequences=True))

model.add(Dropout(0.2))



model.add(LSTM(units=50))

model.add(Dropout(0.2))



model.add(Dense(units = 1))



model.compile(optimizer = 'adam', loss = 'mean_squared_error')



model.fit(features_set, labels, epochs = 100, batch_size = 32)
test_inputs = new_data[len(new_data) - len(valid) - 60:].values

test_inputs = test_inputs.reshape(-1,1)

test_inputs = scaler.transform(test_inputs)



test_features = []

for i in range(60, test_inputs.shape[0]):

    test_features.append(test_inputs[i-60:i, 0])

    

test_features = np.array(test_features)

test_features = np.reshape(test_features, (test_features.shape[0], test_features.shape[1], 1))



predictions = model.predict(test_features)

predictions = scaler.inverse_transform(predictions)

#for plotting

train = new_data[:vCount]

valid = new_data[vCount:]



valid['Predictions'] = predictions

plt.plot(train['Close'], label='Closing Price Trend', lw=2)



#plt.plot(valid[['Close','Predictions']], dashes=[3, 2], label='Actual Vs. Predicted Closing Price', lw=2)

plt.plot(valid['Close'], label='Actual Closing Price', lw=1)

plt.plot(valid['Predictions'], dashes=[1, 2], label='Predicted Price', lw=2)



plt.title('[ ' + vTicker + ' ] - ACTUAL Vs. PREDICTED CLOSING PRICE').set_color('green')

plt.ylabel('Share Price (KES.)')

plt.xlabel('Trading Date')



plt.legend()

plt.show()