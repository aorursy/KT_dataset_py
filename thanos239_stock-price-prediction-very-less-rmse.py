# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import IPython
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
fundamentals= pd.read_csv('../input/nyse/fundamentals.csv')
fundamentals.head()
pricesa_data= pd.read_csv('../input/nyse/prices-split-adjusted.csv')
pricesa_data.head()
price_data= pd.read_csv('../input/nyse/prices.csv')
price_data.head()
security_data= pd.read_csv('../input/nyse/securities.csv')
security_data.head()
#Industry-wise tabulation of number of companies given in the fundamentals data
plt.figure(figsize=(15, 6))
ax = sns.countplot(y='GICS Sector', data=security_data)
plt.xticks(rotation=45)
price_data.info()
len(price_data.symbol.unique())
#let's make a function that'll plot the opening and closing chart for the company chosen

def chart(ticker):
    global closing_stock
    global opening_stock
    f, axs = plt.subplots(2,2,figsize=(16,8))
    plt.subplot(212)
    company = price_data[price_data['symbol']==ticker]
    company = company.open.values.astype('float32')
    company = company.reshape(-1, 1)
    opening_stock = company
    plt.grid(True)
    plt.xlabel('Time')
    plt.ylabel(ticker + " open stock prices")
    plt.title('prices Vs Time')
    
    plt.plot(company , 'g')
    
    plt.subplot(211)
    company_close = price_data[price_data['symbol']==ticker]
    company_close = company_close.close.values.astype('float32')
    company_close = company_close.reshape(-1, 1)
    closing_stock = company_close
    plt.xlabel('Time')
    plt.ylabel(ticker + " close stock prices")
    plt.title('prices Vs Time')
    plt.grid(True)
    plt.plot(company_close , 'b')
    
    plt.show()
ticker = input("Enter the ticker of the company you want to see the graph for -")
chart(ticker)
#make a data-frame that'll have details for the company chosen
ticker_data= price_data[price_data['symbol']==ticker]
#this checks the amount of data we've for the company chosen
train_dates=list(ticker_data.date.unique())
print(f"Period : {len(ticker_data.date.unique())} days")
print(f"From : {ticker_data.date.min()} To : {ticker_data.date.max()}")
#importing libraries
import math
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense , BatchNormalization , Dropout , Activation
from keras.layers import LSTM , GRU
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.optimizers import Adam , SGD , RMSprop
data=ticker_data.copy()
data.head()
data.drop(['symbol','open','low','high','volume'],axis=1,inplace=True)
data.head()
data['date'] = pd.to_datetime(data['date'], infer_datetime_format=True)

data.index=data.date
data.drop('date', axis=1,inplace=True)
data.head()
dataset=data.values
#scale the dataset
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
#we'll use 80% of the data as training data 
train = int(len(dataset) * 0.80)
test = len(dataset) - train
print(train, test)
train= dataset[:train]
test = dataset[len(train):]
train.shape
test.shape
#I'll use past two days data to predict the price for next day. I tried a few numbers and this was giving least error. Therefore, I thought of using this.
x_train, y_train = [], []
for i in range(len(train)-2):
    x_train.append(dataset[i:i+2,0])
    y_train.append(dataset[i+2,0])
x_train, y_train = np.array(x_train), np.array(y_train)
x_train.shape
x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))
x_train.shape
x_train.shape
x_test = []
y_test=[]
for i in range(len(test)-2):
    x_test.append(dataset[len(train)-2+i:len(train)+i,0])
    y_test.append(dataset[len(train)+i,0])
x_test = np.array(x_test)
y_test = np.array(y_test)
x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))
x_test.shape
#I've used a LSTM model to predict the stock price. i checked for various other models but this was giving the least error. Therefore, I've used that
model= Sequential([
                   LSTM(256, input_shape=(x_train.shape[1],1), return_sequences=True),
                   Dropout(0.4),
                   LSTM(256),
                   Dropout(0.2),
                   Dense(16, activation='relu'),
                   Dense(1)
])
print(model.summary())
model.compile(loss='mean_squared_error', optimizer=Adam(lr = 0.0005) , metrics = ['mean_squared_error'])
history = model.fit(x_train, y_train, epochs=40 , batch_size = 128, validation_data=(x_test, y_test))
#summarize history for error
plt.plot(history.history['mean_squared_error'])
plt.plot(history.history['val_mean_squared_error'])
plt.title('model mean squared error')
plt.ylabel('Error')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()
#using the model for x_test and then converting the data to normal price using inverse transform
predicted_price= model.predict(x_test)
predicted_price = scaler.inverse_transform(predicted_price)
len(predicted_price)
predicted_price =np.array(predicted_price)
predicted_price.shape
y_test.shape
#checking the score for our data
def model_score(model, X_train, y_train, X_test, y_test):
    trainScore = model.evaluate(X_train, y_train, verbose=0)
    print('Train Score: %.5f MSE (%.2f RMSE)' % (trainScore[0], math.sqrt(trainScore[0])))
    testScore = model.evaluate(X_test, y_test, verbose=0)
    print('Test Score: %.5f MSE (%.2f RMSE)' % (testScore[0], math.sqrt(testScore[0])))
    return trainScore[0], testScore[0]

model_score(model, x_train, y_train , x_test, y_test)
predicted_price[:10]
y_test = y_test.reshape(y_test.shape[0] , 1)
y_test = scaler.inverse_transform(y_test)
y_test[:10]
#comparing the first 10 values of prediction for our data
diff = predicted_price-y_test
diff[:10]
#plotting the courves for the actual test values and the predicted values. 
#The actual values are represented by the blue line and the predicted value by the red line
print("Red - Predicted Stock Prices  ,  Blue - Actual Stock Prices")
plt.rcParams["figure.figsize"] = (15,7)
plt.plot(y_test , 'b')
plt.plot(predicted_price , 'r')
plt.xlabel('Time')
plt.ylabel('Stock Prices')
plt.title('Check the accuracy of the model with time')
plt.grid(True)
plt.show()