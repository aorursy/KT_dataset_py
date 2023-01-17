# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import requests

import matplotlib.pyplot as plt

import json

import math

import pickle

import joblib

import keras

from datetime import datetime

from requests.exceptions import HTTPError

from sklearn.preprocessing import MinMaxScaler



from keras.models import Sequential

from keras.layers import Dense, LSTM



plt.style.use('fivethirtyeight')



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
'''

try:

    response = requests.get('http://api.marketstack.com/v1/eod?access_key=7a247f5ee1c585714c83b74587866efb&symbols=TCS.XNSE&%20date_from=2000-01-01&limit=1000')

    response.raise_for_status()

    # access JSOn content

    jsonResponse = response.json()

    print("Entire JSON response")

    print(jsonResponse)



except HTTPError as http_err:

    print(f'HTTP error occurred: {http_err}')

except Exception as err:

    print(f'Other error occurred: {err}')

    

'''
type(jsonResponse)
jsonResponse.keys()
jsonResponse['data']
jsonResponse['data'][0]['date']
type(jsonResponse['data'][0]['date'])
jsonResponse['data'][0]['date'][:10]
for value in jsonResponse['data']:

    print(value['date'][:10])
df = pd.DataFrame(jsonResponse['data']) 

df
df.head()
df1 = df[['symbol', 'open', 'close', 'adj_close', 'date']]
df1.dtypes
df1['date'] = pd.to_datetime(df1['date'], format='%Y-%m-%d''T''%H:%M:%S+%f')

df1
df1.set_index('date', inplace = True)

df1
df1.sort_values(by='date', ascending=1, inplace=True)
df1.head()
df1.shape
#Visualize the closing price history

plt.figure(figsize=(16,8))

plt.title('Close Price History')

plt.plot(df1['close'])

plt.xlabel('date',fontsize=18)

plt.ylabel('Close Price in Rupees (Rs)',fontsize=18)

plt.show()
#Create a new dataframe with only the 'Close' column

data = df1.filter(['close'])

#Converting the dataframe to a numpy array

dataset = data.values

#Get /Compute the number of rows to train the model on

training_data_len = math.ceil( len(dataset) *.8) 
#Scale the all of the data to be values between 0 and 1 

scaler = MinMaxScaler(feature_range=(0, 1)) 

scaled_data = scaler.fit_transform(dataset)
#Create the scaled training data set 

train_data = scaled_data[0:training_data_len  , : ]

#Split the data into x_train and y_train data sets

x_train=[]

y_train = []

for i in range(28,len(train_data)):

    x_train.append(train_data[i-28:i,0])

    y_train.append(train_data[i,0])
#Convert x_train and y_train to numpy arrays

x_train, y_train = np.array(x_train), np.array(y_train)
#Reshape the data into the shape accepted by the LSTM

x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))
#Build the LSTM network model

model = Sequential()

model.add(LSTM(units=50, return_sequences=True,input_shape=(x_train.shape[1],1)))

model.add(LSTM(units=50, return_sequences=False))

model.add(Dense(units=25))

model.add(Dense(units=1))
#Compile the model

model.compile(optimizer='adam', loss='mean_squared_error')
#Train the model

model.fit(x_train, y_train, batch_size=2, epochs=20)
type(model)
model.save('StockModelTCS.sav')
#./tcs_model

Oldmodel = keras.models.load_model('./StockModelTCS.sav')
#Test data set

#test_data = scaled_data[training_data_len - 60: , : ]

test_data = scaled_data[training_data_len - 28: , : ]

print(test_data)

#Create the x_test and y_test data sets

x_test = []

y_test =  dataset[training_data_len : , : ] #Get all of the rows from index 1603 to the rest and all of the columns (in this case it's only column 'Close'), so 2003 - 1603 = 400 rows of data

for i in range(28,len(test_data)):

    x_test.append(test_data[i-28:i,0])
training_data_len
#Convert x_test to a numpy array 

x_test = np.array(x_test)
#Reshape the data into the shape accepted by the LSTM

x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))
#Getting the models predicted price values

predictions = Oldmodel.predict(x_test) 

predictions = scaler.inverse_transform(predictions)#Undo scaling
#Calculate/Get the value of RMSE

rmse=np.sqrt(np.mean(((predictions- y_test)**2)))

rmse
train = data[:training_data_len]

valid = data[training_data_len:]

valid['Predictions'] = predictions

#valid['date'] = df1['date']

#Visualize the data

plt.figure(figsize=(16,8))

plt.title('Model')

plt.xlabel('date', fontsize=18)

plt.ylabel('Close Price Rupees (Rs)', fontsize=18)

plt.plot(train['close'])

plt.plot(valid[['close', 'Predictions']])

plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')

plt.show()
#Show the valid and predicted prices

valid
type(valid)
userDate = '2020-07-21'

if userDate in valid.index:

    print(valid._get_value(userDate, 'Predictions'))

else:

    print("Sorry we dont have the data")
from datetime import date

today = date.today()

today
userDate = '2020-10-22'
userDate1 =datetime.strptime(userDate, '%Y-%m-%d').date()
type(userDate1)
userDate1 - today
import datetime

#User need prediction after 27 days

PreviousDate = datetime.date.today() - datetime.timedelta(days=27)

PreviousDate
PreviousDate1 = PreviousDate.strftime("%Y-%m-%d")

while PreviousDate1 not in valid.index:

        PreviousDate = PreviousDate - datetime.timedelta(days=1)

        PreviousDate1 = PreviousDate.strftime("%Y-%m-%d")

        print("Sorry we dont have the data of ",PreviousDate )

if PreviousDate1 in valid.index:

    print(valid._get_value(PreviousDate1, 'close'))

PreviousDate1  
PreviousDate
Days = today - PreviousDate
Days