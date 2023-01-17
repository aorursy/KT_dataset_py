# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train=pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')
train
train=train.dropna(axis=1)
missing_col=set(test)-set(train)
test=test.drop(columns=missing_col,axis=1)
xtrain=train.copy()
xtest=test.copy()
ytrain=xtrain['Target']
ytrain=ytrain.as_matrix()
xtrain=xtrain.drop(columns=['Id','Target','idhogar','dependency','edjefe','edjefa'])
xtest=xtest.drop(columns=['Id','idhogar','dependency','edjefe','edjefa'])
import xgboost as xgb

model=xgb.XGBClassifier(eta=0.2)
model.fit(xtrain,ytrain)

ypred=model.predict(xtest)
ytrain
from keras import models
from keras import layers
from keras import optimizers
from keras.layers import Dropout
from keras.utils import to_categorical

ytrain=to_categorical(ytrain)

model_d=models.Sequential()
model_d.add(layers.Dense(128,activation='relu',input_shape=(xtrain.shape[1],)))
model_d.add(Dropout(0.2))
model_d.add(layers.Dense(128,activation='relu'))
model_d.add(Dropout(0.2))
model_d.add(layers.Dense(5,activation='softmax'))

rmsprop=optimizers.RMSprop(lr=0.01)

model_d.compile(optimizer=rmsprop,loss='categorical_crossentropy',metrics=['accuracy'])

model_d.fit(xtrain,ytrain,epochs=32,batch_size=32)

ypred_d=model_d.predict(xtest)
submission=pd.read_csv('../input/sample_submission.csv')
submission['Target']=ypred

submission.to_csv('sample_submission.csv',index=False)
ypred_d
prediction_deep=[]
from numpy import argmax

for x in ypred_d:
    prediction_deep.append(np.argmax(x))
submission_d=pd.read_csv('../input/sample_submission.csv')
submission_d['Target']=prediction_deep

submission_d.to_csv('sample_submission_deeplearning.csv',index=False)