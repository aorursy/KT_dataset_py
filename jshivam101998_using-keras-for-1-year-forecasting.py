import keras

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline
df=pd.read_csv('../input/for-simple-exercises-time-series-forecasting/Alcohol_Sales.csv',index_col='DATE',parse_dates=True)
df.columns=['sales']
df.head(10)
df.index.freq='MS'
df.plot(figsize=(12,7))
from statsmodels.tsa.seasonal import seasonal_decompose

decompose=seasonal_decompose(df)
decompose.plot();
len(df)
325-12
train=df.iloc[:313]

test=df.iloc[313:]
from sklearn.preprocessing import MinMaxScaler

scaler=MinMaxScaler()

scaler.fit(train)
s_train=scaler.transform(train)
from keras.preprocessing.sequence import TimeseriesGenerator

generator=TimeseriesGenerator(s_train,s_train,length=12,batch_size=1)
from keras.models import Sequential

from keras.layers import LSTM

from keras.layers import Dense
model=Sequential()

model.add(LSTM(200,activation='relu',input_shape=(12,1)))

model.add(Dense(1))

model.compile(loss='mse',optimizer='adam')
model.fit_generator(generator,epochs=100)
loss=model.history.history['loss']

plt.plot(range(len(loss)),loss,color='green')
first_eval_batch=s_train[-12:]

first_eval_batch=first_eval_batch.reshape((1,12,1))

c_batch=first_eval_batch

test_pred=[]

future_pred=[]

for i in range(len(test)+12):    

    c_pred=model.predict(c_batch)[0]

    test_pred.append(c_pred)

    c_batch=np.append(c_batch[:,1:,:],[[c_pred]],axis=1)
len(test_pred)
for i in test_pred[12:]:

    future_pred.append(i)
test_pred=test_pred[:12]
test_pred
len(future_pred)
test_fpred=scaler.inverse_transform(test_pred)

future_fpred=scaler.inverse_transform(future_pred)
test['test_prediction']=test_fpred
test
test.plot(figsize=(12,7),label='test',legend=True)
date=pd.date_range('2019-02-01',periods=12,freq='MS')

prediction=pd.DataFrame(data=list(zip(date,future_fpred)),columns=['date','prediction sale'])
prediction.head()
prediction=prediction.set_index('date')

prediction['pred_sale']=future_fpred
prediction
prediction.index.freq='MS'
prediction.drop('prediction sale',axis=1)
df.plot(figsize=(15,10),label='test',legend=True)

prediction['pred_sale'].plot(legend=True,label='predictions')