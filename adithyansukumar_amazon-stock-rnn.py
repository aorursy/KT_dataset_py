import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
df=pd.read_csv('../input/stock-time-series-20050101-to-20171231/AMZN_2006-01-01_to_2018-01-01.csv',index_col='Date',parse_dates=True)
df.head()
df.info()
df.describe()
df.isnull().sum()

sns.distplot(df['Open'])
sns.distplot(df['High'])
sns.distplot(df['Low'])
sns.distplot(df['Close'])
df.plot(figsize=(19,10))
sns.jointplot(x='Open',y='Close',data=df)
sns.jointplot(x='High',y='Low',data=df)
sns.jointplot(x='Open',y='Volume',data=df)
fig,ax=plt.subplots(figsize=(12,9))
sns.scatterplot(df['Volume'],df['High'],data=df,ax=ax)

sns.heatmap(df.corr(),annot=True,ax=ax)
df.drop('Name',axis=1,inplace=True)
len(df)
test_size=21
test_index=len(df)-test_size
train=df.iloc[:test_index]
test=df.iloc[test_index:]
train
test
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
scaled_train=scaler.fit_transform(train)
scaled_test=scaler.transform(test)
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
length=20
generator=TimeseriesGenerator(scaled_train,scaled_train,length=length,batch_size=1)
validation_generator=TimeseriesGenerator(scaled_test,scaled_test,length=length,batch_size=1)

from tensorflow.keras.models  import Sequential
from tensorflow.keras.layers import LSTM,Dense,Dropout
n_features=scaled_train.shape[1]
model=Sequential()
model.add(LSTM(128,activation='relu',input_shape=(length,n_features),return_sequences=True))
model.add(LSTM(64,activation='relu',input_shape=(length,n_features)))
model.add(Dense(32,activation='relu',input_shape=(length,n_features)))
model.add(Dense(scaled_train.shape[1]))
model.compile(optimizer='adam',loss='mse')
    
model.summary()
from tensorflow.keras.callbacks import EarlyStopping
es=EarlyStopping(monitor='val_loss',patience=2)
model.fit_generator(generator,epochs=100,validation_data=validation_generator,callbacks=[es])
losses=pd.DataFrame(model.history.history)
losses.plot()
n_features = scaled_train.shape[1]
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
true_predictions=scaler.inverse_transform(test_predictions)
from sklearn.metrics import mean_squared_error
print("RMSE=",np.sqrt(mean_squared_error(test,true_predictions)))
test.columns
full_scaler = MinMaxScaler()
scaled_full_data = full_scaler.fit_transform(df)
length = 20 # Length of the output sequences (in number of timesteps)
generator = TimeseriesGenerator(scaled_full_data, scaled_full_data, length=length, batch_size=1)
model=Sequential()
model.add(LSTM(128,activation='relu',input_shape=(length,n_features),return_sequences=True))
model.add(LSTM(64,activation='relu',input_shape=(length,n_features)))
model.add(Dense(32,activation='relu',input_shape=(length,n_features)))
model.add(Dense(scaled_train.shape[1]))
model.compile(optimizer='adam',loss='mse')
    
model.fit_generator(generator,epochs=3)
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
forecast=scaler.inverse_transform(forecast)
df
forecast_index = pd.date_range(start='2017-12-29',periods=periods,freq='D')
forecast_df = pd.DataFrame(data=forecast,index=forecast_index,
                           columns=df.columns)
df.plot()
forecast_df.plot()
forecast_df


ax = df.plot()
forecast_df.plot(ax=ax)
