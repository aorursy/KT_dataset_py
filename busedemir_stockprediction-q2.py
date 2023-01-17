# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
# linear algebra
import numpy as np
# data processing, CSV file I/O (e.g. pd.read_csv)
import pandas as pd
#plt.style.use("seaborn-whitegrid")
import matplotlib.pyplot as plt
# data visualization
import seaborn as sns
%matplotlib inline
from matplotlib import pyplot as plt
from matplotlib import style
# Algorithms
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Import the libraries
import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
df=pd.read_excel("/kaggle/input/stockprediction/StockPrediction(3).xlsx")
#Show the data 
df
df.columns = ['Date', 'Price', 'Opening', 'Dailymax', 'Dailymin', 'Volume', 'Difference']
df.head()
df.Date=pd.to_datetime(df.Date, dayfirst=True)
df = df.sort_values('Date')
df.head()
df.set_index("Date", inplace=True)
df.head()
# The result shows that we have 3545 rows or days the stock price was recorded, and 6 columns.
df.shape
df.isna()
df[df["Volume"].isnull()]
df.isnull().sum()
df[df["Volume"].isnull()]
df.isnull().sum()
df.Volume=df.Volume.fillna(method="ffill")
df.isna().sum()
#Visualize the closing price history
plt.figure(figsize=(16,8))
plt.title('Price History')
plt.plot(df['Price'])
plt.xlabel('Date',fontsize=18)
plt.ylabel('Price USD ($)',fontsize=18)
plt.show()
#Create a new dataframe with only the 'Opening' column
data = df.filter(['Price'])
#Converting the dataframe to a numpy array
dataset = data.values
#Get /Compute the number of rows to train the model on
training_data_len = math.ceil( len(dataset) *.8) 
training_data_len
#Scale the all of the data to be values between 0 and 1 
scaler = MinMaxScaler(feature_range=(0, 1)) 
scaled_data = scaler.fit_transform(dataset)
scaled_data
#Create the scaled training data set 
train_data = scaled_data[0:training_data_len  , : ]
#Split the data into x_train and y_train data sets
x_train=[]
y_train = []
for i in range(60,len(train_data)):
    x_train.append(train_data[i-60:i,0])
    y_train.append(train_data[i,0])
    if i<=60:
        print(x_train)
        print(y_train)
        print()
    

#Convert x_train and y_train to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)
#Reshape the data into the shape accepted by the LSTM
x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))
x_train.shape
#Build the LSTM network model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True,input_shape=(x_train.shape[1],1)))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=25))
model.add(Dense(units=1))
#Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')
#Train the model
model.fit(x_train, y_train, batch_size=1, epochs=1)
test_data = scaled_data[training_data_len - 60: , : ]
#Create the x_test and y_test data sets
x_test = []
y_test =  dataset[training_data_len : , : ] #Get all of the rows from index 3545 to the rest and all of the columns (in this case it's only column 'Close'), so 3545 - 2776 = 769 rows of data
for i in range(60,len(test_data)):
    x_test.append(test_data[i-60:i,0])
#Convert x_test to a numpy array 
x_test = np.array(x_test)
#Reshape the data into the shape accepted by the LSTM
x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))
#Getting the models predicted price values
predictions = model.predict(x_test) 
predictions = scaler.inverse_transform(predictions)#Undo scaling
#Calculate/Get the value of RMSE
rmse=np.sqrt(np.mean(((predictions- y_test)**2)))
rmse
#Plot/Create the data for the graph
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions
#Visualize the data
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Price USD ($)', fontsize=18)
plt.plot(train['Price'])
plt.plot(valid[['Price', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()
#Show the valid and predicted prices
valid
from sklearn.metrics import mean_squared_error, mean_squared_log_error, mean_absolute_error
print("MSE:"+str(mean_squared_error(predictions,y_test)))
print("RMSE:"+str(np.sqrt(mean_squared_error(predictions,y_test))))
print("MSLE:"+str(mean_squared_log_error(predictions,y_test)))
print("RMSLE:"+str(np.sqrt(mean_squared_error(predictions,y_test))))
print("MAE:"+str(mean_squared_error(predictions,y_test)))

df_price=pd.read_excel("/kaggle/input/stockprediction/StockPrediction(3).xlsx")
df_price.columns = ['Date', 'Price', 'Opening', 'Dailymax', 'Dailymin', 'Volume', 'Difference']
df_price.head()
df_price.Date=pd.to_datetime(df_price.Date, dayfirst=True)
df_price = df_price.sort_values('Date')
df_price.head()
df_price.set_index("Date", inplace=True)
df_price.head()
df_price.isna()
df_price[df_price["Volume"].isnull()]
df_price.Volume=df_price.Volume.fillna(method="ffill")
df_price.isna().sum()
#Create a new dataframe
new_df = df_price.filter(['Price'])
#Get teh last 60 day closing price 
last_60_days = new_df[-60:].values
#Scale the data to be values between 0 and 1
last_60_days_scaled = scaler.transform(last_60_days)
#Create an empty list
X_test = []
#Append teh past 60 days
X_test.append(last_60_days_scaled)
#Convert the X_test data set to a numpy array
X_test = np.array(X_test)
#Reshape the data
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
#Get the predicted scaled price
pred_price = model.predict(X_test)
#undo the scaling 
pred_price = scaler.inverse_transform(pred_price)
print(pred_price)
#Get the quote
df_price2=pd.read_excel("/kaggle/input/stockprediction/StockPrediction(3).xlsx")
df_price2.columns = ['Date', 'Price', 'Opening', 'Dailymax', 'Dailymin', 'Volume', 'Difference']
print(df_price2['Price'])
sd=pd.read_excel("/kaggle/input/stockprediction/StockPrediction(3).xlsx")
sd.columns = ['Date', 'Price', 'Opening', 'Dailymax', 'Dailymin', 'Volume', 'Difference']
sd.head()
sd.Date=pd.to_datetime(sd.Date, dayfirst=True)
sd = sd.sort_values('Date')
sd.head()
sd.set_index("Date", inplace=True)
sd.head()
sd.Volume=sd.Volume.fillna(method="ffill")
sd.isna().sum()
#Visualize the closing price history
plt.figure(figsize=(16,8))
plt.title('Price History')
plt.plot(sd['Price'])
plt.xlabel('Date',fontsize=18)
plt.ylabel('Price USD ($)',fontsize=18)
plt.show()
sd.drop(labels= ["Opening", "Dailymax", "Dailymin", "Volume", "Difference"],axis=1,inplace=True)
sd.head()
sd.shift(1)
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(sd)
sd_diff=sd.diff(periods=1)
#integrated of price 1, denoted by d (for diff), one of the parameter of ARIMA model
sd_diff=sd_diff[1:]
sd_diff.head()
plot_acf(sd_diff)
sd_diff.plot(figsize=(20,5), title="Diff")
X1=sd.values
train1=X1[0:2658] #2658 data as train data
test1=X1[2658:] #887 data as test
predictions1=[]
test1.size
from statsmodels.tsa.ar_model import AR
from sklearn.metrics import mean_squared_error
model_ar=AR(train1)
model_ar_fit=model_ar.fit()
predictions1=model_ar_fit.predict(start=2658, end=3545)
test1
plt.plot(test1)
plt.plot(predictions1, color='red')
sd.plot()
from statsmodels.tsa.arima_model import ARIMA
#p,d,q p=periods taken for autoregresive model
#d-> Integrated price, difference
#q period in moving average model
model_arima=ARIMA(train1, order=(2,1,0))
model_arima_fit=model_arima.fit()
predictions1=model_arima_fit.forecast(steps=887)[0]
predictions1
plt.plot(test1)
plt.plot(predictions1, color='red')
print(model_arima_fit.aic)
import itertools
p=d=q=range(0,5)
pdq=list(itertools.product(p,d,q))
pdq
for param in pdq:
    try:
        model_arima=ARIMA(train1, order=param)
        model_arima_fit=model_arima.fit()
        print(model_arima_fit.aic)
    except:
        continue
from sklearn.metrics import mean_squared_error, mean_squared_log_error, mean_absolute_error
print("MSE:"+str(mean_squared_error(predictions1,test1)))
print("RMSE:"+str(np.sqrt(mean_squared_error(predictions1,test1))))
print("MSLE:"+str(mean_squared_log_error(predictions1,test1)))
print("RMSLE:"+str(np.sqrt(mean_squared_error(predictions1,test1))))
print("MAE:"+str(mean_squared_error(predictions1,test1)))
from sklearn.metrics import mean_squared_error, mean_squared_log_error, mean_absolute_error
print("MSE:"+str(mean_squared_error(predictions,y_test)))
print("RMSE:"+str(np.sqrt(mean_squared_error(predictions,y_test))))
print("MSLE:"+str(mean_squared_log_error(predictions,y_test)))
print("RMSLE:"+str(np.sqrt(mean_squared_error(predictions,y_test))))
print("MAE:"+str(mean_squared_error(predictions,y_test)))