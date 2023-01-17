# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#importing the stocks data for apple from Jan 2014 to Oct 2020
import matplotlib.pyplot as plt
%matplotlib inline
df = pd.read_csv("/kaggle/input/apple-stocks/AAPL_Stocks.csv",index_col='Date',infer_datetime_format=True)
df1 = pd.read_csv("/kaggle/input/apple-stocks/AAPL_Stocks.csv")
#plotting opening and closing prices for the Apple stocks data
df1.plot.line(x="Date",y=["Open","Close"],rot=40,linestyle="--",marker='o',markersize=5)
plt.legend(prop={'size':12})
plt.xlabel("Date")
plt.ylabel("Opening and Closing Prices in Dollars")
plt.title("Opening and cLosing Prices")
#splitting the data into testing and traing set
df.iloc[-60]
train_data = df.loc['2014-01-02':'2020-09-01']
test_data = df.loc['2020-09-01':]
#normalizing the data and fitting on the training data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(train_data)
scaled_train = scaler.transform(train_data)
scaled_test = scaler.transform(test_data)
#importing important libraries to create LSTM model
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM
length = 1
batch_size = 1
generator = TimeseriesGenerator(scaled_train,scaled_train,length=length,batch_size=batch_size)
#build our LSTM model to train the data
model = Sequential()
model.add(LSTM(50,input_shape=(length,scaled_train.shape[1])))
model.add(Dense(scaled_train.shape[1]))
model.compile(optimizer="adam",loss="mse")
model.summary()
#creating a validation generator using callbacks and training the model
from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor="val_loss",patience=2)
validation_generator = TimeseriesGenerator(scaled_test,scaled_test,length=length,batch_size=batch_size)

model.fit_generator(generator,epochs=20,validation_data = validation_generator)
#plotting val loss and training loss
losses = pd.DataFrame(model.history.history)
losses.plot()
#now let's have a look at first evaluation batch
first_eval_batch = scaled_train[-length:]
first_eval_batch = first_eval_batch.reshape((1,length,scaled_train.shape[1]))
model.predict(first_eval_batch)
#now lets predict the data
n_features = scaled_train.shape[1]
test_predictions=[]
first_eval_batch = scaled_train[-length:]
current_batch = first_eval_batch.reshape((1,length,n_features))
for i in range(len(test_data)):
    current_pred = model.predict(current_batch)[0]
    test_predictions.append(current_pred)
    current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)
#Having a look at the predictions of test data
true_predictions = scaler.inverse_transform(test_predictions)
true_predictions = pd.DataFrame(data=true_predictions)
predicted_df = pd.DataFrame(data=true_predictions)
predicted_df = predicted_df.rename(columns={0:'Open_Predicted',1:'High_Predicted',2:'Low_Predicted',3:'Close_Predicted',4:'AdjClose_Predicted',5:'Volume_Predicted'})
# as we can see from above predictions that the predicted data is somewhat near to the real data, so now let's proceed to forecasting the data
scaler_new = MinMaxScaler()
scaled_full_data = scaler_new.fit_transform(df)
scaled_full_data
#now we will define our generator object
length = 1
batch_size = 1
n_features = scaled_train.shape[1]
generator = TimeseriesGenerator(scaled_full_data,scaled_full_data,length=length,batch_size=1)
#let's build our model
#build our LSTM model to train the data
model = Sequential()
model.add(LSTM(50,input_shape=(length,scaled_train.shape[1])))
model.add(Dense(scaled_train.shape[1]))
model.compile(optimizer="adam",loss="mse")
model.summary()
model.fit_generator(generator,epochs=20)
#plotting val loss and training loss
losses = pd.DataFrame(model.history.history)
losses.plot()
len(test_data)
#n_features = train.shape[1]
forecast=[]
periods = 32
first_eval_batch = scaled_train[-length:]
current_batch = first_eval_batch.reshape((1,length,n_features))
for i in range(len(test_data)):
    current_pred = model.predict(current_batch)[0]
    forecast.append(current_pred)
    current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)
test_data
#printing the forecast results
forecast = scaler.inverse_transform(forecast)
forecast
#now let's store the forecast results in form of a dataframe
forecast_index = pd.date_range(start='2020-10-16',periods=periods,freq='D')
forecast_df = pd.DataFrame(data=forecast,index=forecast_index)
forecast_df = forecast_df.rename(columns={0:'Open_Future',1:'High_Future',2:'Low_Future',3:'Close_Future',4:'AdjClose_Future',5:'Volume_Future'})
forecast_df.to_csv('StocksForecasting.csv',sep=',')
test = df1.iloc[-32:]
test = test.reset_index()
test = test.drop('index',axis=1)
test_predicted = pd.concat([test,predicted_df],axis=1)
#now let's plot the test data and predicted for Open values on same graph,we can see after plotting the graph as our plotting the
#graph that since our neural network is very simple in the end the values of true and predicted reach to same level
test_predicted.plot.line(x="Date",y=["Open","Open_Predicted"],rot=40,linestyle="--",marker='o',markersize=8)
plt.legend(prop={'size':12})
plt.xlabel("Date")
plt.ylabel("Opening Prices")
plt.title("Opening Prices for test and predicted data in Dollars")