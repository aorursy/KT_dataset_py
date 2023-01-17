# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np
import pandas as pd

data = pd.read_csv("../input/data.csv")

train_encoded = pd.read_csv("../input/train_encoded.csv")
# df=pd.read_csv('data.csv')

df=pd.read_csv("../input/data.csv")
# y=df['label']

np.random.seed(42)

df
df.rename(columns={'UNRATE': 'target'}, inplace=True)

date=df['sasdate']

df=df.drop(columns='sasdate')

col=df.columns[df.isna().any()].tolist()

df=df.drop(columns=col)
df
n_val=1
from sklearn.preprocessing import MinMaxScaler

def create_dataset(dataset,target_index, look_back=20):

    dataX, dataY = [], []

    for i in range(len(dataset) - look_back - 1-n_val):

        #print(i)

        a = dataset[i:(i + look_back)]

        dataX.append(a)

        b = dataset[(i+look_back):(i + look_back+n_val),target_index]

        dataY.append(b)

        #print(dataY)

    return np.array(dataX), np.array(dataY)



# y_scaler = MinMaxScaler(feature_range=(0, 1))

# t_y = df['target'].values.astype('float32')

# yy=t_y

# t_y = np.reshape(t_y, (-1, 1))

# y_scaler = y_scaler.fit(t_y)

# df['target']=y_scaler.transform(t_y)

# Scale and create datasets

target_index = df.columns.tolist().index('target')

print(target_index)

dataset = df.values.astype('float32')



y_scaler = MinMaxScaler(feature_range=(0, 1))

t_y = df['target'].values.astype('float32')

yy=t_y

t_y = np.reshape(t_y, (-1, 1))

y_scaler = y_scaler.fit(t_y)



# Scale the data

scaler = MinMaxScaler(feature_range=(0, 1))

dataset = scaler.fit_transform(dataset)



#Create y_scaler to inverse it later



#print(y)

X, y = create_dataset(dataset,target_index, look_back=24)

# y_scaler = MinMaxScaler(feature_range=(0, 1))

# y = y_scaler.fit_transform(y)

# X_scaler= MinMaxScaler(feature_range=(0, 1))

# X=X_scaler.fit_transform(X)

print(y.shape)

print(X.shape)

#y = y[:, target_index]

#print(y[0])

train_size = int(len(X) * 0.95)

trainX = X[:train_size]

trainY = y[:train_size]

testX = X[train_size:]

testY = y[train_size:]
trainX.shape


sum=0

sum1=0

for i in range(len(yy)):

    if(i>0):

        sum+=abs(yy[i]-yy[i-1])

        sum1+=(yy[i]-yy[i-1])**2

print(sum/(len(yy)-1))

print(sum1/(len(yy)-1))
yy
y[1,:]
print(y.shape)

print(X.shape)

X[0,0]

y=y.reshape(-1,1)

print(y.shape)
import keras

import tensorflow

from keras import Model
from keras.models import Sequential

from keras.layers import *

from keras.regularizers import l2

# model = Sequential()

# model.add(Dense(12, input_dim=20,activation='relu',kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))

# model.add(Dropout(0.2))

# model.add(Dense(4,activation='relu',kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))

# model.add(Dropout(0.2))

# model.add(Dense(1,activation='linear'))

# rms=keras.optimizers.RMSprop(lr=0.001, rho=0.9)

# model.compile(optimizer=rms,

#               loss='mse')

model = Sequential()



# model.add(TimeDistributed(Conv1D(filters=16, kernel_size=2, activation='relu', padding='same'), input_shape=(None, n_length, n_features)))

# model.add(TimeDistributed(MaxPooling1D(pool_size=2)))

# model.add(TimeDistributed(Flatten()))

# model.add(

#     Bidirectional(LSTM(10, input_shape=(X.shape[1], X.shape[2]),

#                        return_sequences=True),

#                   merge_mode='sum',

#                   weights=None,

#                   input_shape=(X.shape[1], X.shape[2])))

model.add(LSTM(30, input_shape=(X.shape[1], X.shape[2]),

                       return_sequences=False))

model.add(Dropout(0.2))

# model.add(LSTM(15, return_sequences=False))

# model.add(Dropout(0.3))



# model.add(LSTM(4, return_sequences=False))

model.add(Dense(30, kernel_initializer='uniform', activation='relu'))

model.add(Dense(n_val, kernel_initializer='uniform', activation='linear'))

rms=keras.optimizers.RMSprop(lr=0.0001, rho=0.9)

model.compile(loss='mean_squared_error', optimizer=rms, metrics=['mae', 'mse'])

print(model.summary())
sum=0

for i in range(len(y)):

    if(i>0):

        sum+=(y[i]-y[i-1])**2

sum/(len(y)-1)
from keras.callbacks import LearningRateScheduler

import keras.backend as K



def scheduler(epoch):

    if epoch%20==0 and epoch!=0:

        lr = K.get_value(model.optimizer.lr)

        K.set_value(model.optimizer.lr, lr*.95)

        print("lr changed to {}".format(lr*.95))

    return K.get_value(model.optimizer.lr)

lr_decay = LearningRateScheduler(scheduler)

filepath="./../weights-{epoch:02d}-{val_mae:.5f}.hdf5"

from keras.callbacks import LearningRateScheduler

from keras.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint(filepath, monitor='val_mae', verbose=1, save_best_only=True, mode='min')

callbacks_list = [checkpoint,lr_decay]

model.fit(trainX,trainY,validation_data=[testX,testY],epochs=2000,batch_size=64,callbacks=callbacks_list)
# from keras.models import load_model

# model=load_model('./../save_3/weights-09-0.01477.hdf5') 
y=y.reshape(len(X),n_val)
import matplotlib.pyplot as plt

plt.plot(y_scaler.inverse_transform(model.predict(X)))

plt.plot(y_scaler.inverse_transform(y))

#plt.savefig('lstm_train_encoded_zoomed.png')

plt.show()
trainX.shape
pred=(model.predict(X))

pred.shape

#y=y.reshape(len(X)-n_val,n_val)

np.mean(np.abs(y_scaler.inverse_transform(y)-y_scaler.inverse_transform(pred)))

plt.plot(y_scaler.inverse_transform(y)[666:701,:],label='actual')

plt.plot(y_scaler.inverse_transform(pred)[666:701,:],label='predicted')

plt.legend()

plt.show()

#y_scaler.inverse_transform(model.predict(X)[696,:].reshape(-1,1))

#y_scaler.inverse_transform(y_scaler.inverse_transform(y[696,:].reshape(-1,1)))
testPred=model.predict(testX)

print(np.mean(np.abs(y_scaler.inverse_transform(testY)-y_scaler.inverse_transform(testPred))))

trainPred=model.predict(trainX)

print(np.mean(np.abs(y_scaler.inverse_transform(trainY)-y_scaler.inverse_transform(trainPred))))

import matplotlib.pyplot as plt

plt.plot(y_scaler.inverse_transform(model.predict(trainX)),label='true')

plt.plot(y_scaler.inverse_transform(trainY),label='predicted')

plt.legend()

#plt.savefig('lstm_test.png')

#plt.legend()

plt.show()



