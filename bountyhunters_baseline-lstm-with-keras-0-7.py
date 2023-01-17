# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

dataPath = "/kaggle/input/m5-forecasting-accuracy/"

timesteps = 14

startDay = 350
dt = pd.read_csv(dataPath + "/sales_train_evaluation.csv")

dt.head(3)

print(dt.info())
#To reduce memory usage

def downcast_dtypes(df):

    float_cols = [c for c in df if df[c].dtype == "float64"]

    int_cols = [c for c in df if df[c].dtype in ["int64", "int32"]]

    df[float_cols] = df[float_cols].astype(np.float32)

    df[int_cols] = df[int_cols].astype(np.int16)

    return df
#Reduce memory usage and compare with the previous one to be sure

dt = downcast_dtypes(dt)
print(dt.info())
#Take the transpose so that we have one day for each row, and 30490 items' sales as columns

dt = dt.T    

dt.head(8)
#Remove id, item_id, dept_id, cat_id, store_id, state_id columns

dt = dt[6 + startDay:]

dt.head(5)
calendar = pd.read_csv(dataPath + "/calendar.csv")

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

daysBeforeEventTest = daysBeforeEvent[1941:1969]

#"daysBeforeEvent" will be used for training as a feature.

daysBeforeEvent = daysBeforeEvent[startDay:1941]



#Before concatanation with our main data "dt", indexes are made same and column name is changed to "oneDayBeforeEvent"

daysBeforeEvent.columns = ["oneDayBeforeEvent"]

daysBeforeEvent.index = dt.index



dt = pd.concat([dt, daysBeforeEvent], axis = 1)



dt.columns
#Feature Scaling

#Scale the features using min-max scaler in range 0-1

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range = (0, 1))

dt_scaled = sc.fit_transform(dt)
X_train = []

y_train = []

for i in range(timesteps, 1941 - startDay):

    X_train.append(dt_scaled[i-timesteps:i])

    y_train.append(dt_scaled[i][0:30490]) 

    #İmportant!! if extra features are added (like oneDayBeforeEvent) 

    #use only sales values for predictions (we only predict sales) 

    #this is why 0:30490 columns are choosen
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



# Adding a third LSTM layer and some Dropout regularisation

layer_3_units=400

regressor.add(LSTM(units = layer_3_units, return_sequences = True))

regressor.add(Dropout(0.2))



# Adding a third LSTM layer and some Dropout regularisation

layer_3_units=400

regressor.add(LSTM(units = layer_3_units))

regressor.add(Dropout(0.2))



# Adding the output layer

regressor.add(Dense(units = 30490))



# Compiling the RNN

regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')



# Fitting the RNN to the Training set

epoch_no=32

batch_size_RNN=44

regressor.fit(X_train, y_train, epochs = epoch_no, batch_size = batch_size_RNN)



inputs= dt[-timesteps:]

inputs = sc.transform(inputs)
X_test = []

X_test.append(inputs[0:timesteps])

X_test = np.array(X_test)

predictions = []



for j in range(timesteps,timesteps + 28):

#X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    predicted_stock_price = regressor.predict(X_test[0,j - timesteps:j].reshape(1, timesteps, 30491))

    testInput = np.column_stack((np.array(predicted_stock_price), daysBeforeEventTest[0][1941 + j - timesteps]))

    X_test = np.append(X_test, testInput).reshape(1,j + 1,30491)

    predicted_stock_price = sc.inverse_transform(testInput)[:,0:30490]

    predictions.append(predicted_stock_price)
import time



submission = pd.DataFrame(data=np.array(predictions).reshape(28,30490))



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



currentDateTime = time.strftime("%d%m%Y_%H%M%S")



submission.to_csv("submission.csv", index=False)

submission.head()