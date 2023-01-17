#import package
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import datetime as dt
import matplotlib.dates as mdates

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense

import warnings
warnings.filterwarnings("ignore")
#read dataset
raw = pd.read_csv('../input/CryptocoinsHistoricalPrices.csv')
raw.head()
#extract data of Ripple from the original dataset
XRP = raw[raw['coin']=='XRP'][['Date','Close']]
#sort data by date
XRP = XRP.sort_values(by='Date', ascending=True)
XRP.head()
#generate the list of close price and convert to 2D array
close_price = XRP.Close.as_matrix()
close_price = np.reshape(close_price,(-1,1))

#generate the list of date and convert to 2D array
date = XRP.Date.as_matrix()
date = np.reshape(date,(-1,1))
#look back 7 days' (shift = 7) close price to predict the close price of the 8th day
shift = 7

#generate x and y dataset
def process_price(dt,shift):
    x,y = [],[]
    length = len(dt) - shift - 1
    for i in range(length):
        x.append(dt[i:shift+i,0])
        y.append(dt[shift+i,0])
    return np.array(x),np.array(y)

#generate date dataset
def process_date(dt,shift):
    time_stamp = []
    length = len(dt) - shift - 1
    for i in range(length):
        time_stamp.append(dt[shift+i,0])
    return np.array(time_stamp)

x,y = process_price(close_price,shift)
date = process_date(date,shift)
#set the amount of train data = 80% of total data
train_ratio = 0.8
interface = int(train_ratio*len(XRP))

#generate train and test datasets
x_train = x[:interface]
x_test = x[interface:]

y_train = y[:interface]
y_test = y[interface:]

date_train = date[:interface]
date_test = date[interface:]
#reshape x and y datasets to 2D array before normalization
x_train = np.reshape(x_train,(-1,1))
x_test = np.reshape(x_test,(-1,1))

y_train = np.reshape(y_train,(-1,1))
y_test = np.reshape(y_test,(-1,1))

#reshape date dataset to 1D array and convert to list
date_train = np.reshape(date_train ,-1,).tolist()
date_test = np.reshape(date_test ,-1,).tolist()
#normalize x and y dataset and reshape back to 3D array
scaler = MinMaxScaler()

scaler.fit(x_train)
norm_x_train = scaler.transform(x_train)
norm_x_train = np.reshape(norm_x_train,(-1,shift,1))

scaler.fit(x_test)
norm_x_test = scaler.transform(x_test)
norm_x_test = np.reshape(norm_x_test,(-1,shift,1))

scaler.fit(y_train)
norm_y_train = scaler.transform(y_train)
norm_y_train = np.reshape(norm_y_train,(-1,1,1))

scaler.fit(y_test)
norm_y_test = scaler.transform(y_test)
norm_y_train = np.reshape(norm_y_train,-1)
#generate LSTM model
run = Sequential()
#add LSTM layer with 256 nodes
run.add(LSTM(units = 256,input_shape = (shift,1),return_sequences = False))
run.add(Dense(1))

#use adam as optimizer to minimize mean square error (mse)
run.compile(optimizer = 'adam',loss = 'mse')

#fit train dataset with the model
run.fit(norm_x_train,norm_y_train,epochs = 100,batch_size = 10, shuffle = True)

#feed norm_x_test dataset to predict norm_y_test
prediction = run.predict(norm_x_test)
prediction = scaler.inverse_transform(prediction)
#convert type of date dateset from string to datetime
date_train = [dt.datetime.strptime(date,'%Y-%m-%d').date() for date in date_train]
date_test = [dt.datetime.strptime(date,'%Y-%m-%d').date() for date in date_test]
#calculate RMSE of the predicted close price from norm_y_test
RMSE = math.sqrt(mean_squared_error(y_test, prediction))
RMSE = round(RMSE, 4)
RMSE = str(RMSE)
RMSE = 'RMSE =  ' + RMSE
#plot figure of close price
plt.style.use('ggplot')

fig1 = plt.figure(figsize=(12,8),dpi=100,)

ax1 = fig1.add_subplot(111)
#plot train dataset against date
ax1.plot(date_train,y_train, linewidth=3.0,color='silver',label='Train dataset')
#convert normalized prices of y_test back to thier original value and plot against date
ax1.plot(date_test,scaler.inverse_transform(norm_y_test.reshape(-1,1)),linewidth=3.0, color='midnightblue',label='Test dataset')
#use the same scale for norm_y_test to modify the prediction result and plot against date
ax1.plot(date_test,prediction,linewidth=2.0,color='maroon',label='Prediction')

ax1.set_ylabel('Close Price (USD)',fontsize=14, color='black')
ax1.set_xlabel('Date',fontsize=14, color='black')
ax1.set_facecolor('white')
ax1.legend(fontsize=12,edgecolor='black',facecolor='white',borderpad=0.75)
ax1.tick_params(colors='black',labelsize=12)
ax1.spines['bottom'].set_color('black')
ax1.spines['top'].set_color('black')
ax1.spines['left'].set_color('black')
ax1.spines['right'].set_color('black')
ax1.grid(color='lightgray', linestyle=':', linewidth=1)

#expand the graph of test dataset and predicted prices
inset = plt.axes([0.2, 0.25, 0.45, 0.45], facecolor='lightgrey')
inset.plot(date_test,scaler.inverse_transform(norm_y_test.reshape(-1,1)),linewidth=3.0, color='midnightblue',label='Test dataset')
inset.plot(date_test,prediction,linewidth=2.0,color='maroon',label='Prediction')

inset.tick_params(colors='black',labelsize=12)
inset.spines['bottom'].set_color('black')
inset.spines['top'].set_color('black')
inset.spines['left'].set_color('black')
inset.spines['right'].set_color('black')
inset.grid(color='white', linestyle=':', linewidth=1)
inset.text(0.6, 0.8,RMSE,transform = inset.transAxes, fontsize = 12)

plt.show()
