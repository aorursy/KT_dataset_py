##stock thai prediciton by Dr. Sompote Youwi
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")
%matplotlib inline

# For reading stock data from yahoo
from pandas_datareader.data import DataReader

# For time stamps
from datetime import datetime
#Get the stock quote
stock_name='MINT.BK'
df = DataReader(stock_name, data_source='yahoo', start='2012-01-01', end=datetime.now())

#plotdata
plt.figure(figsize=(16,8))
plt.title('Close Price History')
plt.plot(df['Close'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price Baht (B)', fontsize=18)
plt.show()
#output data predict
y_time=5
#Create a new dataframe with only the 'Close column
data = df.filter(['Close'])
data2=df.filter(['Volume'])
#Convert the dataframe to a numpy array
dataset = data.values
dataset2 = data2.values
#Get the number of rows to train the model on
training_data_len = int(np.ceil( len(dataset) * .7 ))
training_data_len2 = int(np.ceil( len(dataset) * .7 ))
training_data_len
#Scale the data
from sklearn.preprocessing import MinMaxScaler, StandardScaler

scaler = MinMaxScaler(feature_range=(0,1))
scaler2 = StandardScaler()
scaled_data2 = scaler2.fit_transform(dataset2)
scaled_data = scaler.fit_transform(dataset)

scaled_data


#Create the training data set
#Create the scaled training data set
train_data = scaled_data[0:int(training_data_len), :]
#Split the data into x_train and y_train data sets
x_train = []
y_train = []
datab=60
for i in range(datab, len(train_data)):
    x_train.append(train_data[i-datab:i, 0])
    y_train.append(train_data[i-y_time:i, 0])
    '''if i<= (datab+1):
        print(x_train)
        print(y_train)
        print()'''
#volume
train_data2 = scaled_data2[0:int(training_data_len), :]
#Split the data into x_train and y_train data sets 
#second feature
x_train2 = []
datab=60
for i in range(datab, len(train_data2)):
    x_train2.append(train_data2[i-datab:i, 0])
    #if i<= (datab+1):
       #print(x_train2)
        #print()


# Convert the x_train and y_train to numpy arrays 
x_train, y_train = np.array(x_train), np.array(y_train)

#Reshape the data
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_train=np.dstack((x_train, x_train2))
print(x_train.shape)
print(y_train.shape)

from keras.models import Sequential
from keras.layers import Dense, LSTM

#Build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape= (x_train.shape[1],x_train.shape[2] )))
model.add(LSTM(50, return_sequences= False))
model.add(Dense(25))
model.add(Dense(y_time))
# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

#Train the model
model.fit(x_train, y_train, batch_size=1, epochs=4)
#Create the testing data set
#Create a new array containing scaled values from index 1543 to 2002 
test_data = scaled_data[training_data_len - datab: , :]
test_data2 = scaled_data2[training_data_len - datab: , :]

#Create the data sets x_test and y_test
x_test = []
x_test2=[]
y_test = dataset[training_data_len:, :]
for i in range(datab, len(test_data)):
    x_test.append(test_data[i-datab:i, 0])
for i in range(datab, len(test_data2)):
    x_test2.append(test_data[i-datab:i, 0])
# Convert the data to a numpy array
x_test = np.array(x_test)
x_test2 = np.array(x_test2)
# Reshape the data
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))
x_test=np.dstack((x_test, x_test2))
# Get the models predicted price values 
predictions1 = model.predict(x_test)
predictions = scaler.inverse_transform(predictions1)
# Get the root mean squared error (RMSE)
rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
predicte=predictions[:,0]
print('RSME',rmse)

# Plot the data
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predicte
# Visualize the data
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price THB (B)', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()
model.save('stock.h5')
#predict next 5 days

from tensorflow import keras
from pandas_datareader.data import DataReader
stock_name='MINT.BK'
model = keras.models.load_model('stock.h5')
#Get the quote
apple_quote = DataReader(stock_name, data_source='yahoo', start='2012-01-01', end='2020-11-07')
#Create a new dataframe
new_df = apple_quote.filter(['Close'])
new_df2=apple_quote.filter(['Volume'])
#Get teh last 60 day closing price 

last_60_days = new_df[-60:]
last_60_days_v = new_df2[-60:]
#Scale the data to be values between 0 and 1
last_60_days_scaled = scaler.transform(last_60_days)
last_60_days_scaled_v = scaler2.transform(last_60_days_v)
#Create an empty list
X_test = []
X_test2 = []
Y_pre =  []
  
  
#Append teh past 60 days
X_test.append(last_60_days_scaled)
X_test2.append(last_60_days_scaled_v)
#Convert the X_test data set to a numpy array
X_test = np.array(X_test)
X_test2 = np.array(X_test2)
#Reshape the data
X_test2 = np.reshape(X_test2, (X_test2.shape[0], X_test2.shape[1]))#
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]))
X_testr=np.stack((X_test, X_test2),axis=2)
#Get the predicted scaled price
pred_price = model.predict(X_testr)
pred_price = scaler.inverse_transform(pred_price)
print(pred_price)

'''from tensorflow import keras
from pandas_datareader.data import DataReader
stock_name='MINT.BK'
model = keras.models.load_model('stock.h5')
#Get the quote
apple_quote = DataReader(stock_name, data_source='yahoo', start='2012-01-01', end='2020-06-07')
#Create a new dataframe
new_df = apple_quote.filter(['Close'])
new_df2=apple_quote.filter(['Volume'])
#Get teh last 60 day closing price 
pre_time=300

for i in range(1,pre_time):
  last_60_days = new_df[-60:]
  last_60_days_v = new_df2[-60:]
  #Scale the data to be values between 0 and 1
  last_60_days_scaled = scaler.transform(last_60_days)
  last_60_days_scaled_v = scaler2.transform(last_60_days_v)
  #Create an empty list
  X_test = []
  X_test2 = []
  Y_pre =  []
  
  
#Append teh past 60 days
  X_test.append(last_60_days_scaled)
  X_test2.append(last_60_days_scaled_v)
#Convert the X_test data set to a numpy array
  X_test = np.array(X_test)
  X_test2 = np.array(X_test2)
#Reshape the data
  X_test2 = np.reshape(X_test2, (X_test2.shape[0], X_test2.shape[1]))
  X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]))
  X_testr=np.stack((X_test, X_test2),axis=2)
#Get the predicted scaled price
  pred_price = model.predict(X_testr)
  pred_price = scaler.inverse_transform(pred_price)[0,0]
  new_df=new_df.append({'Close': pred_price}, ignore_index=True)

'''

'''train = data[:training_data_len]
valid = data[training_data_len:]
# Visualize the data
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price THB (B)', fontsize=18)
plt.plot(new_df['Close'])
#plt.plot(new_df['Close'])

#plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()'''
data = new_df['Close']
new_df['Close']
