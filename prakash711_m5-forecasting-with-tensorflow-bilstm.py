# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

dataPath = '../input/m5-forecasting-accuracy/'

timesteps = 14

startDay = 350

dt = pd.read_csv(dataPath + "/sales_train_validation.csv")

dt.head(3)

print(dt.info())
def reduction_mem(df):

    float_cols = [c for c in df if df[c].dtype == 'float64']

    int_cols = [c for c in df if df[c].dtype in ['int64', 'int32']]

    df[float_cols] = df[ float_cols].astype(np.float16)

    df[int_cols] = df[int_cols].astype(np.int16)

    return df

    
#Reduce memory usage and compare with the previous one to be sure

dt = reduction_mem(dt)
print(dt.info())
#Take the transpose so that we have one day for each row, and 30490 items' sales as columns

dt = dt.T

dt.head(8)
#Remove id, item_id, dept_id, cat_id, store_id, state_id columns

dt = dt[6 + startDay:]

dt.head(5)
cal_data = pd.read_csv(dataPath + 'calendar.csv')
#Create dataframe with zeros for 1969 days in the calendar

daysBeforeEvent = pd.DataFrame(np.zeros((1969,1)))

# "1" is assigned to the days before the event_name_1. Since "event_name_2" is rare, it was not added.

for x,y in cal_data.iterrows():

    if((pd.isnull(cal_data["event_name_1"][x])) == False):

           daysBeforeEvent[0][x-1] = 1 

            #if first day was an event this row will cause an exception because "x-1".

            #Since it is not i did not consider for now.



   
#"calendar" won't be used anymore. 

del cal_data

#"daysBeforeEventTest" will be used as input for predicting (We will forecast the days 1913-1941)

daysBeforeEventTest = daysBeforeEvent[1913:1941]

#"daysBeforeEvent" will be used for training as a feature.

daysBeforeEvent = daysBeforeEvent[startDay:1913]



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

for i in range(timesteps, 1913 - startDay):

    X_train.append(dt_scaled[i-timesteps:i])

    y_train.append(dt_scaled[i][0:30490]) 

    #İmportant!! if extra features are added (like oneDayBeforeEvent) 

    #use only sales values for predictions (we only predict sales) 

    #this is why 0:30490 columns are choosen
del dt_scaled

#Convert to np array to be able to feed the  BiLSTM model

X_train = np.array(X_train, dtype = 'float16')

y_train = np.array(y_train, dtype = 'float32')

print(X_train.shape)

print(y_train.shape)
#for the GPU

import tensorflow as tf

model = tf.keras.models.Sequential([

        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True), input_shape= (np.array(X_train).shape[1], np.array(X_train).shape[2])),

        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16)),

        tf.keras.layers.Dense(30490),

    ])

model.summary()

model.compile(optimizer = 'adam', loss = 'mean_squared_error')

lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss',factor=0.5, patience=3,

                                                    verbose=1,

                                                    min_lr=0.000001,

                                                   )
#model training for GPU

model.fit(X_train, y_train, epochs = 50, batch_size = 10,callbacks=[lr_scheduler])
inputs= dt[-timesteps:]

inputs = sc.transform(inputs)
X_test = []

X_test.append(inputs[0:timesteps])

X_test = np.array(X_test)

predictions = []



for j in range(timesteps,timesteps + 28):

#X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    predicted_stock_price = model.predict(X_test[0,j - timesteps:j].reshape(1, timesteps, 30491))

    testInput = np.column_stack((np.array(predicted_stock_price), daysBeforeEventTest[0][1913 + j - timesteps]))

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