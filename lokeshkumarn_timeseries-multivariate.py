# This Python 3 environment comes with many helpful analytics libraries installed
import numpy as np
import pandas as pd
from datetime import datetime

import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Dense,LSTM,GRU

import os
print(os.listdir("../input"))
def parse(x):
    return datetime.strptime(x, '%Y %m %d %H')
data = pd.read_csv('../input/ChinaPollution.csv',index_col=0,date_parser=parse,parse_dates=[['year','month','day','hour']])
data.drop(columns=['No'],axis=1,inplace=True)
data.columns = ['pollution', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']
data.index.name='date'
data['pollution'].fillna(0.0,inplace=True)
data = data[24:]
data.head()
data.info()
data.columns
fig,ax = plt.subplots(7,1,figsize=(20,15))
for i,column in enumerate([col for col in data.columns if col != 'wnd_dir']):
    data[column].plot(ax=ax[i])
    ax[i].set_title(column)
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg
labelEncoder = LabelEncoder()
data.iloc[:,4] = labelEncoder.fit_transform(data.iloc[:,4])
values = data.values
print(values.shape)
values = values.astype('float32')
values
series_to_supervised(values,1,1)
reframed = series_to_supervised(values,1,1)
# drop columns we don't want to predict
reframed.drop(reframed.columns[[9,10,11,12,13,14,15]], axis=1, inplace=True)
reframed.head()
scaler = MinMaxScaler(feature_range=(0,1))
scaled = scaler.fit_transform(values)
reframed = series_to_supervised(scaled,1,1)
# drop columns we don't want to predict
reframed.drop(reframed.columns[[9,10,11,12,13,14,15]], axis=1, inplace=True)
reframed.head()
values = reframed.values
n_train = 365*24
train = values[:n_train]
test = values[n_train:]
trainX,trainY = train[:,:-1],train[:,-1]
testX,testY = test[:,:-1],test[:,-1]
print(trainX.shape,trainY.shape,testX.shape,testY.shape)
trainX = trainX.reshape(trainX.shape[0],1,trainX.shape[1])
testX = testX.reshape(testX.shape[0],1,testX.shape[1])
print(trainX.shape)
print(testX.shape)
stop_noimprovement = EarlyStopping(patience=10)
model = Sequential()
model.add(LSTM(50,input_shape=(trainX.shape[1],trainX.shape[2]),dropout=0.2))
model.add(Dense(1))
model.compile(loss="mae",optimizer="adam")
history= model.fit(trainX,trainY,validation_data=(testX,testY),epochs=100,verbose=2,callbacks=[stop_noimprovement],shuffle=False)
plt.plot(history.history['loss'],label='train')
plt.plot(history.history['val_loss'],label='test')
plt.legend()
plt.show()
predicted = model.predict(testX)
testXRe = testX.reshape(testX.shape[0],testX.shape[2])
predicted = np.concatenate((predicted,testXRe[:,1:]),axis=1)
predicted.shape
predicted = scaler.inverse_transform(predicted)
testY = testY.reshape(len(testY),1)
testY.shape
testY = np.concatenate((testY,testXRe[:,1:]),axis=1)
testY = scaler.inverse_transform(testY)
pd.DataFrame(testY)
np.sqrt(mean_squared_error(testY[:,0],predicted[:,0]))
result = pd.concat([pd.Series(predicted[:,0]),pd.Series(testY[:,0])],axis=1)
result.columns = ['thetahat','theta']
result['diff'] = result['thetahat'] - result['theta']
result



















