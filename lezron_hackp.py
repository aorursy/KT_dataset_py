# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import tensorflow as tf

from keras.layers import Dense,Dropout

from keras.models import Sequential

import keras

from sklearn.model_selection import train_test_split

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
np.random.seed(50)
df = pd.read_csv('../input/train.csv')

df.replace(to_replace=-200,value=0,inplace=True)

df.columns
sns.heatmap(df.corr())
X = df[['AH', 'C6H6(GT)', 'NOx(GT)', 'PT08.S1(CO)', 'PT08.S2(NMHC)', 'PT08.S3(NOx)',

       'PT08.S4(NO2)', 'PT08.S5(O3)', 'RH']]

y = df[['T']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=42)
mean = X_train.mean(axis = 0)

std = X_train.std(axis = 0)

X_train = (X_train - mean)/std

X_test = (X_test - mean)/std

X_train.shape
(X_train.shape[0])
def build_model():

    model = keras.Sequential([

        Dense(64,activation = tf.nn.relu,input_shape = (X_train.shape[1],)),

        Dense(32,activation = tf.nn.relu),

        Dense(8,activation = tf.nn.relu),

        Dense(1)

    ])

    

    optimizer = tf.train.RMSPropOptimizer(0.001)

    model.compile(loss = 'mse',optimizer = optimizer,metrics = ['mae'])

    return model

model = build_model()

model.summary()

    
EPOCHS = 400

class PrintDot(keras.callbacks.Callback):

    def on_epoch_end(self,epoch,logs):

        if(epoch % 100 == 0): print(' ')

        print('.',end = ' ')

early_stop = keras.callbacks.EarlyStopping(monitor = 'val_loss',patience = 40)

    

y_pred = model.fit(X_train,y_train,batch_size=40,epochs = EPOCHS,validation_split = 0.2,shuffle=True,verbose =0,callbacks = [early_stop,PrintDot()])
[loss, mae] = model.evaluate(X_test,y_test,verbose = 0)

mae
test = pd.read_csv('../input/test.csv')

#test.replace(to_replace=-200,value=0,inplace=True)
X_testing = test[['AH', 'C6H6(GT)','NOx(GT)', 'PT08.S1(CO)', 'PT08.S2(NMHC)', 'PT08.S3(NOx)',

       'PT08.S4(NO2)', 'PT08.S5(O3)', 'RH']]

X_testing = (X_testing - mean)/std
y_pred = model.predict(X_testing).flatten()

k = y_pred[87]
y_pred[87]
submission = pd.DataFrame({

    'Date_Time':test['Date_Time'],

    'T':y_pred

})
submission.to_csv("submit.csv", index = False)