import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
df=pd.read_csv('../input/retail-sales-forecasting/mock_kaggle.csv',index_col='data',parse_dates=True)
df.head()
df.columns
df.plot(figsize=(12,8))
df.tail

test_size=21
test_ind=len(df)-test_size
test_ind
train=df.iloc[:test_ind]
train
test=df.iloc[test_ind:]
test
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
scaled_train=scaler.fit_transform(train)
scaled_test=scaler.transform(test)
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
length=19
generator=TimeseriesGenerator(scaled_train,scaled_train,length=length,batch_size=1)
x,y=generator[0]
x
y
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM,Dropout
scaled_train.shape
n_features=scaled_train.shape[1]
model=Sequential()
model.add(LSTM(100,activation='relu',input_shape=(length,n_features),return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(64,activation='relu',input_shape=(length,n_features)))
model.add(Dropout(0.2))
model.add(Dense(32,activation='relu',input_shape=(length,n_features)))
model.add(Dense(n_features))
model.compile(optimizer='adam',loss='mse')
model.summary()
validation_generator=TimeseriesGenerator(scaled_test,scaled_test,length=length,batch_size=1)
from tensorflow.keras.callbacks import EarlyStopping
es=EarlyStopping(monitor='val_loss',patience=2)
model.fit_generator(generator,epochs=50,validation_data=validation_generator,callbacks=[es])
losses=pd.DataFrame(model.history.history)
losses.plot()
first_eval_batch = scaled_train[-length:]
first_eval_batch = first_eval_batch.reshape(1,length,n_features)
model.predict(first_eval_batch)
scaled_test[0]
test_predictions = []

first_eval_batch = scaled_train[-length:]
current_batch = first_eval_batch.reshape((1, length,scaled_train.shape[1]))

for i in range(len(test)):
    
    # get prediction 1 time stamp ahead ([0] is for grabbing just the number instead of [array])
    current_pred = model.predict(current_batch)[0]
    
    # store prediction
    test_predictions.append(current_pred) 
    
    # update batch to now include prediction and drop first value
    current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)
true_predictions = scaler.inverse_transform(test_predictions)
from sklearn.metrics import mean_squared_error
np.sqrt(mean_squared_error(test,true_predictions))
full_scaler = MinMaxScaler()
scaled_full_data = full_scaler.fit_transform(df)
length = 365 # Length of the output sequences (in number of timesteps)
generator = TimeseriesGenerator(scaled_full_data, scaled_full_data, length=length, batch_size=1)
model=Sequential()
model.add(LSTM(100,activation='relu',input_shape=(length,n_features),return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(64,activation='relu',input_shape=(length,n_features)))
model.add(Dropout(0.2))
model.add(Dense(32,activation='relu',input_shape=(length,n_features)))
model.add(Dense(n_features))
model.compile(optimizer='adam',loss='mse')
model.fit_generator(generator,epochs=50)
forecast = []
# Replace periods with whatever forecast length you want
periods = 21

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
forecast_index=pd.date_range(start='2016-07-31',periods=periods,freq='D')
forecast_df = pd.DataFrame(data=forecast,index=forecast_index,
                           columns=[df.columns])
forecast_df
ax = df.plot()
forecast_df.plot(ax=ax,figsize=(12,9))
df.plot()
forecast_df.plot()
