import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math

import matplotlib.pyplot as plt
import datetime as dt
import urllib.request, json
import os

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
# sales data set for train
train_sales_df = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv')
# calendar
calendar_df = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/calendar.csv')
# sell price
sell_prices_df = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sell_prices.csv')
#Let's make a list of date columns date_col = [d1,d2,d3,d4...]
date_col = [col for col in train_sales_df if col.startswith('d_')]
hobbies_sales_df = train_sales_df[train_sales_df.id.eq('HOBBIES_1_001_CA_1_validation')]
#Create sales df where days are a column
hobbies_sales_df = pd.melt(hobbies_sales_df, id_vars = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"])

hobbies_sales_df.columns = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id", "d", "sales"]

# Join sales data with calendar
hobbies_sales_df = pd.merge(hobbies_sales_df, calendar_df, left_on = 'd', right_on = 'd')
hobbies_sales_df.shape
hobbies_sales_df.sort_values('date')

hobbies_sales_df.head()
# fix random seed for reproducibility
np.random.seed(7)
dataframe = hobbies_sales_df.loc[:,'sales']
dataset = dataframe.values
dataset = dataset.astype('float32')
# Split data in train and test
#train_data = hobbies_sales_matrix[:1600]
#test_data = hobbies_sales_matrix[1600:]
dataframe
#plt.plot(dataset)
#plt.show()

import plotly.express as px
fig = px.scatter(x=np.arange(dataset.shape[0]), y=dataset)
fig.show()



dataset = dataset[900:,]
dataset_original = dataset
dataset_original.shape
# normalize the dataset
dataset = (dataset - np.min(dataset_original))/np.ptp(dataset_original)
plt.plot(dataset)
plt.show()
# Train and test data
train = dataset[:800]
test = dataset[800:]
# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    return np.array(dataX), np.array(dataY)
# reshape into X=t and Y=t+1
look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
# create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=20, batch_size=40, verbose=2)
# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
# invert predictions
trainPredict = (np.ptp(dataset_original)*trainPredict)+np.min(dataset_original)
trainY = (np.ptp(dataset_original)*trainY)+np.min(dataset_original)
testPredict = (np.ptp(dataset_original)*testPredict)+np.min(dataset_original)
testY = (np.ptp(dataset_original)*testY)+np.min(dataset_original)
# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY, trainPredict))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY, testPredict))
print('Test Score: %.2f RMSE' % (testScore))
# shift train predictions for plotting
trainPredictPlot = dataset
trainPredictPlot[:] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back] = np.squeeze(trainPredict)
# shift test predictions for plotting
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:] = np.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1] = np.squeeze(testPredict)
# plot baseline and predictions
plt.plot(dataset_original)
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()
print(sum(trainPredict))
print(sum(trainY))

print(sum(testPredict))
print(sum(testY))
hobbies_sales_df.sort_values('date')

hobbies_sales_df.head()
hobbies_sales_df.info()
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
train_dataset= pd.DataFrame()
train_dataset['ds'] = pd.to_datetime(hobbies_sales_df["date"][900:1600])
train_dataset['y']=hobbies_sales_df["sales"][900:1600]
train_dataset.head(2)
prophet_basic = Prophet()
prophet_basic.fit(train_dataset)
future= prophet_basic.make_future_dataframe(periods=300)
future.tail(2)
forecast=prophet_basic.predict(future)
fig1 =prophet_basic.plot(forecast)