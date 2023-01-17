import os 
print(os.listdir('../input'))
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv("../input/stocks/GOOGL_daily.csv")
df = df[::-1]          # reverse index because top column is the most recent price 
df = df.reset_index() # reset its index
df.head()
df.shape
df.describe()
data = df.loc[:,'1. open'] 
plt.subplots(figsize = (25,10))
plt.ylabel('Price', fontsize=16)
plt.xlabel('Date', fontsize=16)
plt.title('Google open-price chart')
plt.plot(data)
plt.show()
print(data)
dataset = np.array(data)
print(dataset)
training_data_len = int(np.ceil(len(dataset) * .8))
print(training_data_len)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset.reshape(-1,1))
scaled_data
train_x = []
train_y = []
train_data = scaled_data[0:int(training_data_len),:]

for i in range(60,len(train_data)):
    train_x.append(train_data[i-60:i,0])
    train_y.append(train_data[i,0])
    if i<= 61:
        print(train_x)
        print(train_y)
train_x,train_y = np.array(train_x) , np.array(train_y)
train_x.shape
train_x = np.reshape(train_x,(train_x.shape[0], train_x.shape[1], 1))
train_x.shape
import tensorflow.keras as keras
from keras.models import Sequential
from keras.layers import Dense,Dropout,LSTM
# Build Model
model = Sequential()
model.add(LSTM(units = 50 , return_sequences=True , input_shape = (train_x.shape[1],1)))
model.add(LSTM(units = 50 , return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units= 50 , return_sequences = False))
model.add(Dense(25))
model.add(Dense(1))
keras.utils.plot_model(model = model , to_file = 'StockLSTM.png')
model.summary()
model.compile(optimizer='adam' , loss='mean_squared_error')
num_epoch = 50
model.fit(train_x,train_y,batch_size=32 , epochs=num_epoch)
test_data = scaled_data[training_data_len - 60:,:]
test_x = []
test_y = dataset[training_data_len:]
for i in range(60,len(test_data)):
    test_x.append(test_data[i-60:i,0])
test_x = np.array(test_x)
test_x = np.reshape(test_x,(test_x.shape[0] , test_x.shape[1] , 1))
test_x.shape
prediction = model.predict(test_x)
prediction = scaler.inverse_transform(prediction)
# Get the root mean squared error (RMSE)
rmse = np.sqrt(np.mean(((prediction - test_y) ** 2)))
rmse
train = data[:training_data_len]
actual = data[training_data_len:]
valid=pd.DataFrame()
valid['Actual'] = actual
valid['Prediction'] = prediction
# Visualize the data
plt.figure(figsize=(16,8))
plt.title('Model', fontsize=16)
plt.xlabel('Date', fontsize=16)
plt.ylabel('Open Price USD ($)', fontsize=16)
plt.plot(train)
plt.plot(valid[['Actual','Prediction']])
plt.legend(['Train', 'Actual', 'Prediction'], loc='lower right')
plt.show()
# Visualize the data
plt.figure(figsize=(16,8))
plt.title('Model', fontsize=16)
plt.xlabel('Date', fontsize=16)
plt.ylabel('Open Price USD ($)', fontsize=16)
plt.plot(valid[['Actual','Prediction']])
plt.legend(['Actual', 'Prediction'], loc='lower right')
plt.show()
