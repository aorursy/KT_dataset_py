# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
# print(os.listdir())
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
# Any results you write to the current directory are saved as output.
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from keras import optimizers
from sklearn import preprocessing
from keras.models import load_model

callbacks = [EarlyStopping(monitor='val_mean_absolute_error', patience=8),
             ModelCheckpoint(filepath='best_model.h5', monitor='val_mean_absolute_error', save_best_only=True)]

X = train.iloc[:,3:-1]
Y = train.iloc[:,-1]
X_train, X_test, Y_train, Y_test =  train_test_split(X,Y,test_size = 0.2, random_state = 8)
scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

best_model = Sequential()
best_model.add(Dense(80,input_dim=X_train.shape[1],activation='selu'))
best_model.add(Dense(160,activation='selu'))
best_model.add(Dense(320,activation='selu'))
best_model.add(Dropout(0.1))
best_model.add(Dense(160,activation='selu'))
best_model.add(Dense(80,activation='selu'))
best_model.add(Dense(40,activation='selu'))
best_model.add(Dense(20,activation='selu'))
best_model.add(Dense(1,activation='sigmoid'))
adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-8, amsgrad=False)
best_model.compile(optimizer=adam, loss='mse', metrics=['mae'])
history = best_model.fit(X_train, Y_train, epochs=80,
          validation_data=(X_test, Y_test),batch_size=10000,
         callbacks=callbacks)
    
# activation = selu and add dropout? increase batch size?
best_model = load_model('best_model.h5')
predict_y = best_model.predict(scaler.transform(test.iloc[:,3:]))
submission = pd.concat([test['Id'],pd.Series(np.clip(predict_y.flatten(), a_min=0, a_max=1),name='winPlacePerc')],axis=1)
submission.to_csv('submission.csv', index=False)