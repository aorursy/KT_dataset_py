import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt
df= pd.read_csv('../input/real-manufacturing-and-trade-inventories-2020/INVCMRMT.csv', index_col='DATE', parse_dates=True)

df['INVCMRMT']=df['INVCMRMT'].astype(int)

df.index.freq= 'MS'

df['INVCMRMT'].plot(figsize=(12,6)) 
len(df) #We will grap the last year for forecasting



train= df.iloc[:265]

test = df.iloc[265:]
from sklearn.preprocessing import MinMaxScaler

scaler= MinMaxScaler()

scaler.fit(train)

scaled_train= scaler.transform(train)

scaled_test= scaler.transform(test)
len(test)
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

length= 12 #batch size should be smaller than test size

generator= TimeseriesGenerator(scaled_train,scaled_train,length=length,batch_size=1)
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense,LSTM

n_features=1 # 1 variable and 1 col 'INVCMRMT'



model= Sequential()

model.add(LSTM(400,activation='relu', input_shape=(length,n_features))) #input shape of batch size

model.add(Dense(1))

model.compile(optimizer='adam',loss='mse')

model.summary()
from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor='val_loss',patience=2) #patience is number of epochs with no improvment

validation_generator= TimeseriesGenerator(scaled_test,scaled_test,length=length,batch_size=1)

model.fit_generator(generator,epochs=10,validation_data=validation_generator,callbacks=early_stop)
losses= pd.DataFrame(model.history.history)

losses.plot()
test_predictions = []



first_eval_batch = scaled_train[-length:]

current_batch = first_eval_batch.reshape((1, length, n_features))



for i in range(len(test)):

    

    # get prediction 1 time stamp ahead ([0] is for grabbing just the number instead of [array])

    current_pred = model.predict(current_batch)[0]

    

    # store prediction

    test_predictions.append(current_pred) 

    

    # update batch to now include prediction and drop first value

    current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)
# IGNORE WARNINGS

true_predictions = scaler.inverse_transform(test_predictions)

test['Predictions'] = true_predictions

test.plot(figsize=(12,8))
test_predictions = []



first_eval_batch = scaled_train[-length:]

current_batch = first_eval_batch.reshape((1, length, n_features))



for i in range(len(test)):

    

    # get prediction 1 time stamp ahead ([0] is for grabbing just the number instead of [array])

    current_pred = model.predict(current_batch)[0]

    

    # store prediction

    test_predictions.append(current_pred) 

    

    # update batch to now include prediction and drop first value

    current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)
true_predictions = scaler.inverse_transform(test_predictions)

test['Predictions'] = true_predictions
test
test.plot(figsize=(12,8))
full_scaler= MinMaxScaler()

scaled_full_data=full_scaler.fit_transform(df)
length=12

generator=TimeseriesGenerator(scaled_full_data,scaled_full_data,length=length,batch_size=1)

model= Sequential()

model.add(LSTM(400,activation='relu', input_shape=(length,n_features))) #input shape of batch size

model.add(Dense(1))

model.compile(optimizer='adam',loss='mse')

model.fit_generator(generator,epochs=6)
forecast = []

# Replace periods with whatever forecast length you want

periods = 12



first_eval_batch = scaled_full_data[-length:]

current_batch = first_eval_batch.reshape((1, length, n_features))



for i in range(periods):

    

    # get prediction 1 time stamp ahead ([0] is for grabbing just the number instead of [array])

    current_pred = model.predict(current_batch)[0]

    

    # store prediction

    forecast.append(current_pred) 

    

    # update batch to now include prediction and drop first value

    current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)
forecast = scaler.inverse_transform(forecast)
df
forecast_index = pd.date_range(start='2020-08-01',periods=periods,freq='MS')

forecast_df = pd.DataFrame(data=forecast,index=forecast_index,columns=['Forecast'])
forecast_df
ax = df.plot()

forecast_df.plot(ax=ax)
ax = df.plot()

forecast_df.plot(ax=ax)

plt.xlim('2018-08-01','2021-08-01')