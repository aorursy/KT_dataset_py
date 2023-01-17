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
class data_preparation(object):
    def __init__(self,price,time,train,valid,test,shift):
        self.price = price #price dataset
        self.time = time #datetime dataset
        self.train = train #fration of train data
        self.valid = valid #fration of valid data
        self.test = test #fration  of test data
        #train + valid + test = 1
        self.shift = shift #length of historical price using for predicting future price
        self.interface1 = (int(self.train*len(self.price))//self.shift)*self.shift
        self.interface2 = self.interface1 + ((int(self.valid*len(self.price))//self.shift)*self.shift)
        self.end = (len(self.price)//self.shift)*self.shift
              
    #function for generating x or y dataset
    global xy
    def xy(data,shift,str):
        series = []
        length = len(data) - shift - 1
        for i in range(length):
            if str == 'x':
                series.append(data[i:shift+i,0])
            else:
                series.append(data[shift+i,0])
        return np.array(series)
    
    #function for separating train, valid and test dataset
    global allocation    
    def allocation(x_or_y,interface1,interface2,end):
        train = x_or_y[:interface1]
        valid = x_or_y[interface1:interface2]
        test = x_or_y[interface2:end]
        return train, valid, test
        
    #function for generating unnormalized x dataset
    global x
    def x(self):
        x = xy(self.price,self.shift,'x')
        x_train, x_valid, x_test = allocation(x,self.interface1,self.interface2,self.end)
        return x_train, x_valid, x_test

    #function for generating unnormalized y dataset
    global y
    def y(self):
        y = xy(self.price,self.shift,'y')
        y_train, y_valid,y_test = allocation(y,self.interface1,self.interface2,self.end)
        return y_train, y_valid,y_test
        #function for generating unnormalized y dataset
    
    #function for generating unnormalized y dataset for visualization
    def actual_y(self):
        y = xy(self.price,self.shift,'y')
        y_train, y_valid,y_test = allocation(y,self.interface1,self.interface2,self.end)
        return y_train, y_valid,y_test
    
    #normalization function
    global norm
    def norm(pre_norm):
        scaler = MinMaxScaler()
        pre_norm = np.reshape(pre_norm,(-1,1))
        scaler.fit(pre_norm)
        norm = scaler.transform(pre_norm)
        return norm
    
    #function for generating normalized x dataset
    def norm_x(self):
        x_train, x_valid, x_test = x(self)
        norm_x_train = np.reshape(norm(x_train) ,(-1,self.shift,1))        
        norm_x_valid = np.reshape(norm(x_valid) ,(-1,self.shift,1))  
        norm_x_test = np.reshape(norm(x_test) ,(-1,self.shift,1))
        return norm_x_train, norm_x_valid, norm_x_test
    
    #function for generating normalized y dataset
    def norm_y(self):
        y_train, y_valid, y_test = y(self)
        norm_y_train = np.reshape(norm(y_train) ,(-1,1))        
        norm_y_valid = np.reshape(norm(y_valid) ,(-1,1))  
        norm_y_test = np.reshape(norm(y_test) ,(-1,1))  
        return norm_y_train, norm_y_valid, norm_y_test
    
    #function for generating time domain of each dataset
    def timestamp(self):
        time = xy(self.time,self.shift,'date')
        time_train, time_valid,time_test = allocation(time,self.interface1,self.interface2,self.end)
        return time_train, time_valid, time_test
#feed price and time datasets to data_preparation
#set shift = 7 ->look back 7 days' (shift = 7) close price to predict the close price of the 8th day
shift = 7
process = data_preparation(price = close_price, time = date,train = 0.7, valid=0.10, test = 0.20, shift = shift)
#generate normalized x and y dataset
norm_x_train, norm_x_valid,norm_x_test = process.norm_x()
norm_y_train, norm_y_valid,norm_y_test = process.norm_y()
#generate LSTM model
run = Sequential()
#add LSTM layer with 256 nodes
run.add(LSTM(units = 256,input_shape = (shift,1),return_sequences = False))
run.add(Dense(1))

#use adam as optimizer to minimize mean square error (mse)
run.compile(optimizer = 'adam',loss = 'mse')

#fit train dataset with the model
run.fit(norm_x_train,norm_y_train, validation_data = (norm_x_valid,norm_y_valid),epochs = 100,batch_size = 10, shuffle = True)

#feed norm_x_test dataset to predict norm_y_test
prediction = run.predict(norm_x_test)
#generate date and unnormalized y datasets for visualization
y_train, y_valid,y_test = process.actual_y()
date_train, date_valid,date_test = process.timestamp()
#scale y_test dataset
#this scaler will be used to convert the normalized predicted result to actual price
y_test_scaler = MinMaxScaler()
y_test = np.reshape(y_test,(-1,1))
y_test_scaler.fit(y_test)

#scale up the predicted close price
prediction = y_test_scaler.inverse_transform(prediction)
#convert type of date dateset from string to datetime
date_train = [dt.datetime.strptime(date,'%Y-%m-%d').date() for date in date_train]
date_valid = [dt.datetime.strptime(date,'%Y-%m-%d').date() for date in date_valid]
date_test = [dt.datetime.strptime(date,'%Y-%m-%d').date() for date in date_test]
#calculate RMSE of the predicted close price from y_test
RMSE = math.sqrt(mean_squared_error(y_test, prediction))
RMSE = round(RMSE, 4)
RMSE = str(RMSE)
RMSE = 'RMSE =  ' + RMSE
#plot figure of close price
plt.style.use('ggplot')

fig1 = plt.figure(figsize=(12,8),dpi=100,)

ax1 = fig1.add_subplot(111)
#plot train dataset against date
ax1.plot(date_train,y_train, linewidth=3.0,color='lightgray',label='Train dataset')
#plot valid dataset against date
ax1.plot(date_valid,y_valid, linewidth=3.0,color='darkgray',label='Validation dataset')
#plot test dataset against date
ax1.plot(date_test,y_test,linewidth=3.0, color='midnightblue',label='Test dataset')
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
inset.plot(date_test,y_test,linewidth=3.0, color='midnightblue',label='Test dataset')
inset.plot(date_test,prediction,linewidth=2.0,color='maroon',label='Prediction')

inset.tick_params(colors='black',labelsize=12)
inset.spines['bottom'].set_color('black')
inset.spines['top'].set_color('black')
inset.spines['left'].set_color('black')
inset.spines['right'].set_color('black')
inset.grid(color='white', linestyle=':', linewidth=1)
inset.text(0.6, 0.8,RMSE,transform = inset.transAxes, fontsize = 12)

plt.show()