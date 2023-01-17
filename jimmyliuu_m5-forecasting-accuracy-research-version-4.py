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
CalendarDF=pd.read_csv("/kaggle/input/m5-forecasting-accuracy/calendar.csv", header=0)

SalesDF=pd.read_csv("/kaggle/input/m5-forecasting-accuracy/sales_train_evaluation.csv", header=0) #June 1st Dataset

CalendarDF['date'] = pd.to_datetime(CalendarDF.date)



TX_1_Sales = SalesDF[['TX_1' in x for x in SalesDF['store_id'].values]]

TX_1_Sales = TX_1_Sales.reset_index(drop = True)



# Generate MultiIndex for easier aggregration.

TX_1_Indexed = pd.DataFrame(TX_1_Sales.groupby(by = ['cat_id','dept_id','item_id']).sum())



# Aggregate total sales per day for each sales category

Food = pd.DataFrame(TX_1_Indexed.xs('FOODS').sum(axis = 0))

Hobbies = pd.DataFrame(TX_1_Indexed.xs('HOBBIES').sum(axis = 0))

Household = pd.DataFrame(TX_1_Indexed.xs('HOUSEHOLD').sum(axis = 0))



# Merge the aggregated sales data to the calendar dataframe based on date

CalendarDF = CalendarDF.merge(Food, how = 'left', left_on = 'd', right_on = Food.index)

CalendarDF = CalendarDF.rename(columns = {0:'Food'})

CalendarDF = CalendarDF.merge(Hobbies, how = 'left', left_on = 'd', right_on = Hobbies.index)

CalendarDF = CalendarDF.rename(columns = {0:'Hobbies'})

CalendarDF = CalendarDF.merge(Household, how = 'left', left_on = 'd', right_on = Household.index)

CalendarDF = CalendarDF.rename(columns = {0:'Household'})



# Drop dates with null sales data

CalendarDF = CalendarDF.drop(CalendarDF.index[1941:])

CalendarDF.reset_index(drop = True)
# Modify Food data to feed into model



Food.index = CalendarDF.date

foodValues = Food.values



# Normalize data

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range = (0,1))

foodValues = scaler.fit_transform(foodValues)



# Train and Test datasets

foodTrain = foodValues[0:1899]

foodTest = foodValues[1899:1941]
# Create a function to convert array of values into dataset matrix

def create_dataset(dataset, look_back = 1):

    dataX, dataY = [], []

    for i in range(len(dataset) - look_back - 1):

        a = dataset[i:(i+look_back), 0]

        dataX.append(a)

        dataY.append(dataset[i+look_back, 0])

    return np.array(dataX), np.array(dataY)
# Modify train and test datasets

look_back = 1

trainX, trainY = create_dataset(foodTrain, look_back)

testX, testY = create_dataset(foodTest, look_back)



trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))

testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
# Create and fit the LSTM network

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM



model = Sequential()

model.add(LSTM(4, input_shape=(1, look_back)))

model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)
# Make predictions

trainPredict = model.predict(trainX)

testPredict = model.predict(testX)

# Invert predictions to scale

trainPredict = scaler.inverse_transform(trainPredict)

trainY = scaler.inverse_transform([trainY])

testPredict = scaler.inverse_transform(testPredict)

testY = scaler.inverse_transform([testY])
# Calculate root mean squared error

import math

from sklearn.metrics import mean_squared_error



trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))

print('Train Score: %.2f RMSE' % (trainScore))

testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))

print('Test Score: %.2f RMSE' % (testScore))



# Plot

import matplotlib.pyplot as plt



plt.plot(Food['20160411':'20160522'].values)

plt.plot(testPredict)

plt.show()
# Train and Test datasets

foodTrain = Food['20110129':'20160410']

foodTest = Food['20160411':'20160522']



# Simple linear smoothing model

from statsmodels.tsa.holtwinters import Holt



model = Holt(np.asarray(foodTrain.values))

model.index = pd.to_datetime(foodTrain.index)



fit1 = model.fit(smoothing_level=.3, smoothing_slope=.05)

pred1 = fit1.predict(1899, 1940)

pred1DF = pd.DataFrame(pred1, foodTest.index)



# Estimating the model paramaters by maximizing log values

fit2 = model.fit(optimized = True)

pred2 = fit2.predict(1899,1940)

pred2DF = pd.DataFrame(pred2, foodTest.index)



# Uses brute force optimizer to search for good starting values

fit3 = model.fit(use_brute = True)

pred3 = fit3.predict(1899,1940)

pred3DF = pd.DataFrame(pred3,foodTest.index)   



plt.plot(foodTest)

plt.plot(pred1DF)

plt.plot(pred2DF)

plt.plot(pred3DF)



plt.show()