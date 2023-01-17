# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pickle
dataPath = "/kaggle/input/m5-forecasting-accuracy/"

timesteps = 14

startDay = 403
# read data files

calendar = pd.read_csv(dataPath + '/calendar.csv')

sales_train_val = pd.read_csv(dataPath + '/sales_train_validation.csv')

sample_submission = pd.read_csv(dataPath + '/sample_submission.csv')
NUM_ITEMS = sales_train_val.shape[0]  # 30490

DAYS_PRED = sample_submission.shape[1] - 1  # 28
# keep a multiple of DAYS_PRED days (columns) for convenience

ncols = sales_train_val.shape[1] - sales_train_val.shape[1] % DAYS_PRED

sales_train_val = sales_train_val.iloc[:, -ncols:]
# Take the transpose so that we have one day for each row, and 30490 items' sales as columns

sales_train_val = sales_train_val.T
# get seasonal data in same format as sales_train_val

def reformat(fit_forecast):

    fit_forecast = np.concatenate(fit_forecast)

    fit_forecast = fit_forecast.reshape(-1, NUM_ITEMS, order='F')

    fit = fit_forecast[:-DAYS_PRED, :]

    forecast = fit_forecast[-DAYS_PRED:, :]

    return fit, forecast
# load pre-calculate yearly seasonal fit & forecast 

file = open('/kaggle/input/seasonalities/seasonality/yearly_seasonal.pkl', 'rb')

fit_forecast1 = pickle.load(file)

file.close()
# get fit1 and forecast1

fit1, forecast1 = reformat(fit_forecast1)
# remove trend and first seasonal component from sales

sales_train_val = sales_train_val - fit1
# remove first 28-day period as we have no fit

sales_train_val = sales_train_val.iloc[28:, :]
# the first year looks atypical so we will delete it

sales_train_val = sales_train_val.iloc[365:, :]
# load pre-calculated dow seasonality

file = open('/kaggle/input/seasonalities/seasonality/dow_seasonal.pkl', 'rb')

fit_forecast2 = pickle.load(file)

file.close()
# get fit2 and forecast2

fit2, forecast2 = reformat(fit_forecast2)
# remove trend and first seasonal component from sales

sales_train_val = sales_train_val - fit2
# we have no fit for the first day

sales_train_val = sales_train_val.iloc[1:, :]
# load pre-calculated event and snap effects

file = open('/kaggle/input/seasonalities/seasonality/events_snap_dm.pkl', 'rb')

fit_forecast3 = pickle.load(file)

file.close()
# get fit3 and forecast3

fit3, forecast3 = reformat(fit_forecast3)
# remove latest fit from sales

sales_train_val = sales_train_val - fit3
# load pre-calculated day-in-month fit & forecast

file = open('/kaggle/input/seasonalities/seasonality/dayinmonth.pkl', 'rb')

fit_forecast4 = pickle.load(file)

file.close()
# get fit4 and forecast4

fit4, forecast4 = reformat(fit_forecast4)
# remove latest fit from sales

sales_train_val = sales_train_val - fit4
#Create dataframe with zeros for 1969 days in the calendar

daysBeforeEvent = pd.DataFrame(np.zeros((1969,1)))
# "1" is assigned to the days before the event_name_1. Since "event_name_2" is rare, it was not added.

for x,y in calendar.iterrows():

    if((pd.isnull(calendar["event_name_1"][x])) == False):

           daysBeforeEvent[0][x-1] = 1 

            #if first day was an event this row will cause an exception because "x-1".

            #Since it is not i did not consider for now.
#"calendar" won't be used anymore. 

del calendar
#"daysBeforeEventTest" will be used as input for predicting (We will forecast the days 1913-1941)

daysBeforeEventTest = daysBeforeEvent[1913:1941]

#"daysBeforeEvent" will be used for training as a feature.

daysBeforeEvent = daysBeforeEvent[startDay:1913]
#Before concatanation with our main data "dt", indexes are made same and column name is changed to "oneDayBeforeEvent"

daysBeforeEvent.columns = ["oneDayBeforeEvent"]

daysBeforeEvent.index = sales_train_val.index
sales_train_val = pd.concat([sales_train_val, daysBeforeEvent], axis = 1)
#Feature Scaling

#Scale the features using min-max scaler in range 0-1

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range = (0, 1))

dt_scaled = sc.fit_transform(sales_train_val)
X_train = []

y_train = []

for i in range(timesteps, 1913 - startDay):

    X_train.append(dt_scaled[i-timesteps:i])

    y_train.append(dt_scaled[i][0:30490]) 
del dt_scaled
#Convert to np array to be able to feed the LSTM model

X_train = np.array(X_train)

y_train = np.array(y_train)

print(X_train.shape)

print(y_train.shape)
# Importing the Keras libraries and packages

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

from keras.layers import Dropout



# Initialising the RNN

regressor = Sequential()



# Adding the first LSTM layer and some Dropout regularisation

layer_1_units=50

regressor.add(LSTM(units = layer_1_units, return_sequences = True, input_shape = (X_train.shape[1], X_train.shape[2])))

regressor.add(Dropout(0.2))



# Adding a second LSTM layer and some Dropout regularisation

layer_2_units=400

regressor.add(LSTM(units = layer_2_units, return_sequences = True))

regressor.add(Dropout(0.2))



# Adding a third LSTM layer and some Dropout regularisation

layer_3_units=400

regressor.add(LSTM(units = layer_3_units))

regressor.add(Dropout(0.2))



# Adding the output layer

regressor.add(Dense(units = NUM_ITEMS))



# Compiling the RNN

regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')



# Fitting the RNN to the Training set

epoch_no=32

batch_size_RNN=44

regressor.fit(X_train, y_train, epochs = epoch_no, batch_size = batch_size_RNN)
# inputs for prediction

inputs = sales_train_val[-timesteps:]

inputs = sc.transform(inputs)
X_test = []

X_test.append(inputs[0:timesteps])

X_test = np.array(X_test)

predictions = []



for j in range(timesteps,timesteps + 28):

#X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    predicted_stock_price = regressor.predict(X_test[0,j - timesteps:j].reshape(1, timesteps, 30491))

    testInput = np.column_stack((np.array(predicted_stock_price), daysBeforeEventTest[0][1913 + j - timesteps]))

    X_test = np.append(X_test, testInput).reshape(1,j + 1,30491)

    predicted_stock_price = sc.inverse_transform(testInput)[:,0:30490]

    predictions.append(predicted_stock_price)
# reformat predictions and add forecasts

submission = pd.DataFrame(data=np.array(predictions).reshape(28,30490))

submission = submission + forecast1 + forecast2 + forecast3 + forecast4
# reformat submission

submission = submission.T

submission = pd.concat((submission, submission), ignore_index=True)

sample_submission = pd.read_csv(dataPath + "/sample_submission.csv")

idColumn = sample_submission[["id"]]

submission[["id"]] = idColumn  

cols = list(submission.columns)

cols = cols[-1:] + cols[:-1]

submission = submission[cols]

colsdeneme = ["id"] + [f"F{i}" for i in range (1,29)]

submission.columns = colsdeneme

submission.to_csv("submission.csv", index=False)