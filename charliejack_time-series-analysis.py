#Importing stuff

import warnings

warnings.filterwarnings('ignore')

from statsmodels.tools.sm_exceptions import ConvergenceWarning

warnings.simplefilter('ignore', ConvergenceWarning)

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from matplotlib import pylab

import sklearn.metrics

plt.style.use('fivethirtyeight')

%matplotlib inline



import seaborn as sns

%pylab inline

sns.set(style="darkgrid") #Comment out if pyplot style is wanted



import datetime as dt

import tensorflow as tf

import tensorflow.keras as keras

import tensorflow.keras.layers as L

from sklearn.preprocessing import MinMaxScaler



import os

print(f"filename: {os.listdir('../input/financial-markets')}")
#Class to help with formatting printed text

class color:

   BOLD = '\033[1m'

   END = '\033[0m'
data = pd.read_csv('../input/financial-markets/Index2018.csv',

                   parse_dates=['date'],

                   date_parser = (lambda date: dt.datetime.strptime(date,'%d/%m/%Y'))

                   ,index_col='date')      



data.head()
print(color.BOLD + 'Description of data:' + color.END)

print(data.describe())



print(color.BOLD + '\nNumber of Null values:' + color.END)

data.isna().sum()
sns.set(font_scale=(1.5))

data.plot(subplots=True, figsize=(15,12),ylabel =('Close Value'),

                         title = ['spx from 1994-2018','dax from 1994-2018',

                                  'ftse from 1994-2018','nikkei from 1994-2018',])

plt.savefig('stocks.png')

plt.show()
sns.set(font_scale=2.0)

data[['ftse','spx']].pct_change().multiply(100).plot(subplots=True,figsize=(20,8),fontsize=20,ylabel='Gain (%)')

sns.set(font_scale=(1.4))

data['ftse_return'] = data.ftse.diff() 

data['spx_return'] = data.spx.diff()

data['dax_return'] = data.dax.diff()

data['nikkei_return'] = data.nikkei.diff()

data[['ftse_return',

'spx_return',

'dax_return',

'nikkei_return']].plot(subplots=True,figsize=(15,15),fontsize=20,ylabel='Absolute gain',legend=True)



print(color.BOLD + 'Absolute returns for all 4 stocks' + color.END)
format_ftse = data.ftse.divide(data.ftse.iloc[0]).multiply(100)

format_spx = data.spx.divide(data.spx.iloc[0]).multiply(100)

format_nikkei = data.nikkei.divide(data.nikkei.iloc[0]).multiply(100)

format_dax = data.dax.divide(data.dax.iloc[0]).multiply(100)



sns.set(font_scale=(1.9))

format_ftse.plot(fontsize=20,ylabel='Relative value',legend=True)

format_spx.plot(fontsize=20,ylabel='Relative value',legend=True)

format_dax.plot(fontsize=20,ylabel='Relative value',legend=True)

format_nikkei.plot(figsize=(18,8),fontsize=20,ylabel='Relative value',legend=True)
#First lets view just SPX

sns.set(font_scale=(1.3))

data.spx.plot(figsize=(10,6),fontsize=15,ylabel='Close Value')
#We will use statsmodels' seasonal decomposer (multiplicitive)

#So, ftse[t] = Trend[t] * Seasonal[t] * Noise[t]



import statsmodels.api as sm

from pylab import rcParams



rcParams['figure.figsize'] = 11, 9

spx_decomposed = sm.tsa.seasonal_decompose(data.spx,period=400,model='multiplicative') #yearly seasonality



spx_decomposed.plot()

plt.show()
dataset = data.spx #this is the stock data we want

dataset = np.array(dataset.values) #turning into an array



print(color.BOLD + 'dataset peak:\n' + color.END,dataset)



#getting the length of training data, with a 80/20 split (ish, not quite divisible perfectly)

training_len = int(len(dataset)*0.80//1 -15)



test_len = int(len(dataset) - training_len)



print(('\ntraining length: {}\ntest length: {}').format(training_len,test_len))
#Applying scaling using Sklearn's MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,1)) 

scaled_data = scaler.fit_transform(dataset.reshape(-1, 1))



#Now splitting data into train and test

train_data = scaled_data[:training_len,:]

test_data = scaled_data[training_len-50:,:]
x_train = []

y_train = []

x_test = []

window_size = 50



for i in range(window_size, len(train_data)):

    x_train.append(train_data[i-window_size:i, 0])

    y_train.append(train_data[i, 0])

    



for i in range(window_size, len(test_data)):

    x_test.append(test_data[i-window_size:i, 0])

            

#Converting into numpy arrays to feed in model  

x_train = np.array(x_train)

y_train = np.array(y_train)

x_test = np.array(x_test)



#data must be in a 3d array - as we have  amount of data points, window_size, and batch size

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1)) 

x_test = np.reshape(x_test,(x_test.shape[0], x_test.shape[1] , 1))

print('x_train shape: ', np.shape(x_train))

print('x_test_shape', np.shape(x_test))
#Build the LSTM model

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, LSTM, Conv1D , Dropout #Conv1d and Dropout didn't improve this model, so have been left out



model = Sequential()

model.add(LSTM(50, return_sequences=True,input_shape=(x_train.shape[1],1)))

model.add(LSTM(50, return_sequences= False))

model.add(Dense(25,activation='relu'))

model.add(Dense(1))



# Compile the model

model.compile(optimizer='Adam', loss='mean_squared_error')



#Train the model

model.fit(x_train, y_train, batch_size=1, epochs=1)
predictions = model.predict(x_test) #obtaining the models predictions

predictions = scaler.inverse_transform(predictions) #Obtaining actual predictions from the normalised predictions

predictions = predictions[:,0]

print(predictions)
train = data.spx[:training_len]

validation = pd.DataFrame(data.spx[training_len:])

validation['Predictions'] = predictions



sns.set(font_scale=1.5)



fig,(ax1,ax2) = plt.subplots(1,2,figsize=(16,8))

ax2.plot(train)

ax2.plot(validation[['spx','Predictions']])

ax2.legend(['Train', 'Validation', 'Predictions'], loc='lower right')

ax2.set_xlabel('Date', fontsize=16)

ax2.set_ylabel('Close Price USD ($)', fontsize=16)





ax1.plot(validation['spx'],color='darkorange')

ax1.plot(validation['Predictions'],color='green')

ax1.legend(['Validation','Predictions'],loc='lower right')

ax1.set_xlabel('Date', fontsize=16)

ax1.set_ylabel('Close Price USD ($)', fontsize=16)

#'shifting' the data by 1, to get the naive forecasting, and duplicating the previous value to preserve lengths



naive = np.append(validation.spx.values[1:],validation.spx.values[-1])



#plotting

plt.plot(naive-validation.spx.values,label = 'naive')

plt.plot(validation.spx.values-predictions,color='green',label = 'lstm predictions')

plt.legend(fontsize=14)

plt.ylabel('error')

plt.xlabel('data point',fontsize=13)



print('MAE of predictions:',sklearn.metrics.mean_absolute_error(validation.spx.values,predictions))

print('MSE of predictions:',sklearn.metrics.mean_squared_error(validation.spx.values,predictions)**0.5)

print('\nMSE of naive-forecast:',np.mean(np.square((validation.spx.values-naive)))**0.5)