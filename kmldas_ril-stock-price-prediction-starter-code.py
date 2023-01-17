

import math

import pandas_datareader as web

import numpy as np

import pandas as pd

from datetime import datetime

import math





from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential

from keras.layers import Dense, LSTM

import matplotlib

import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')





ril_price= pd.read_csv("../input/reliance-industries-ril-share-price-19962020/Reliance Industries 1996 to 2020.csv")

#Show the data 

ril_price
ril_price=ril_price.dropna()

ril_price
ril_price.info()
ril_price["Date"]=pd.to_datetime(ril_price["Date"], format="%d-%m-%Y")





ril_price["Date"]



ril_price.set_index('Date', inplace=True)

ril_price.info()
ril_price.describe()
#Visualize the closing price history

plt.figure(figsize=(16,8))

plt.title('Reliance Industries Close Price History')

plt.plot(ril_price['Close'])

plt.xlabel('Date',fontsize=18)

plt.ylabel('Close Price INR',fontsize=18)

plt.show()
#Create a new dataframe with only the 'Close' column

data = ril_price.filter(['Close'])

#Converting the dataframe to a numpy array

dataset = data.values

#Get /Compute the number of rows to train the model on

training_data_len = math.ceil( len(dataset) *.8) 