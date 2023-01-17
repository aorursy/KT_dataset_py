# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.models import Sequential
import keras.backend as K
import random
from keras import regularizers
from keras.layers import Reshape, multiply ,Activation, Conv2D, Multiply, Input, MaxPooling2D, SpatialDropout2D,BatchNormalization, Flatten, Dense, Lambda, Dropout, LSTM,CuDNNGRU,CuDNNLSTM,GRU,Conv1D
from keras.optimizers import SGD, Adam, RMSprop
import os
print(os.listdir("../input/"))
data=pd.read_csv('../input/bitstampUSD_1-min_data_2012-01-01_to_2018-11-11.csv', delimiter=',')
data.head(5)
data[data.columns.values] = data[data.columns.values].ffill()
data[data.columns.values]=data[data.columns.values].fillna(0)
def preparedata(data, periodtochange=10, periodtopredict=100):
    data['change']=(data['Weighted_Price']-data['Weighted_Price'].shift(periodtochange))/data['Weighted_Price']
    data['+gain']=(data['Close'].shift(-periodtopredict)-data['Close'])/data['Weighted_Price']*100
    data['-gain']=(-data['Close'].shift(-periodtopredict)+data['Close'])/data['Weighted_Price']*100
    return data
data=preparedata(data)
data[data.columns.values]=data[data.columns.values].fillna(0)
datatrain=data[:1990342]
dataval=data[1990342:-300]
def data_generator(df, batchsize=10240, length=200):
    xsez=np.array(df['change'])
    ysez=np.zeros((len(df),3))
    ysez[:,:2]=np.array(df[['+gain','-gain']])
    sequences=np.zeros((batchsize, length, 1))
    answer=np.zeros((batchsize, 3))
    j=1
    k=0
    while True:
        k=random.randint(1,xsez.shape[0]-length-2)
        seq=xsez[k:k+length]
        ans=ysez[k+length, :]
        sequences[j%batchsize,:,0]=seq
        answer[j%batchsize,:]=ans
        j=j+1
        if j>len(df)-length-2: j=0
        if j%batchsize==0:
            yield sequences, answer
genvalregr=data_generator(dataval, batchsize=10240, length=200)
toval=next(genvalregr)
def Gain_loss(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    toret = -K.mean(y_true_f * y_pred_f)
    return toret
biasl1=0
kernell1=0
biasl2=0.01
kernell2=0.01
model = Sequential()
model.add(Conv1D(16, 32, activation='sigmoid', input_shape=(None, 1), kernel_regularizer=regularizers.l1_l2(l1=kernell1, l2=kernell2) ,bias_regularizer=regularizers.l1_l2(l1=biasl1, l2=biasl2)))
model.add(BatchNormalization())
model.add(GRU(16, kernel_regularizer=regularizers.l1_l2(l1=kernell1, l2=kernell2) ,bias_regularizer=regularizers.l1_l2(l1=biasl1, l2=biasl2)))
model.add(BatchNormalization())
model.add(Dense(32,  activation='sigmoid',kernel_regularizer=regularizers.l1_l2(l1=kernell1, l2=kernell2) ,bias_regularizer=regularizers.l1_l2(l1=biasl1, l2=biasl2)))
model.add(BatchNormalization())
model.add(Dense(32,  activation='sigmoid',kernel_regularizer=regularizers.l1_l2(l1=kernell1, l2=kernell2) ,bias_regularizer=regularizers.l1_l2(l1=biasl1, l2=biasl2)))
model.add(BatchNormalization())
model.add(Dense(3,  activation='softmax',kernel_regularizer=regularizers.l1_l2(l1=kernell1, l2=kernell2) ,bias_regularizer=regularizers.l1_l2(l1=biasl1, l2=biasl2)))
model.compile(loss=Gain_loss, optimizer=Adam(lr=0.001))
model.summary()
from keras.callbacks import LearningRateScheduler
def annealing(x):
    initial_lrate = 0.001
    return initial_lrate/(x+1)*np.random.rand()
lrate = [LearningRateScheduler(annealing)]
K.set_value(model.optimizer.lr, 0.0001)
hist=model.fit_generator(data_generator(datatrain, batchsize=1024, length=200),
    samples_per_epoch = 10, 
    epochs = 10,
    validation_data=(toval[0], toval[1]),
    verbose=1,
    callbacks=lrate,
    )