import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

%matplotlib inline

import os

from sklearn.metrics import log_loss

from sklearn.model_selection import train_test_split



import tensorflow as tf

from tensorflow import keras 

from keras import backend as K

import tensorflow_addons as tfa
train = pd.read_csv('../input/lish-moa/train_features.csv')

test = pd.read_csv('../input/lish-moa/test_features.csv')

train_targets_scored = pd.read_csv('../input/lish-moa/train_targets_scored.csv')

train_targets_nonscored = pd.read_csv('../input/lish-moa/train_targets_nonscored.csv')

submission = pd.read_csv('../input/lish-moa/sample_submission.csv')
print('Number of rows in train data :',train.shape[0])

print('Number of rows in test data :',test.shape[0])
train.shape,test.shape
train.head(10)
test.head(10)
train_targets_scored.head(10)
t1 = train.drop('sig_id',axis=1)

t2 = test.drop('sig_id',axis=1)

targets = train_targets_scored.drop('sig_id',axis=1)

data = pd.concat([t1,t2],axis=0)

data.head()
data_ohe = pd.get_dummies(data)

data_ohe.shape
train_ohe = data_ohe[:train.shape[0]]

test_ohe = data_ohe[train.shape[0]:]

train_ohe.shape,test_ohe.shape
train_ohe.head()
X_train,X_val,Y_train,Y_val = train_test_split(train_ohe,targets,test_size=0.2)

X_train.shape,X_val.shape,Y_train.shape,Y_val.shape
model = tf.keras.models.Sequential([

    keras.layers.Dense(256,activation='relu',input_shape=(X_train.shape[1],)),

    keras.layers.BatchNormalization(),

    keras.layers.Dropout(0.5),

    keras.layers.Dense(206,activation='softmax')

])
model.summary()
model.compile(metrics=['accuracy'],optimizer='adam',loss='binary_crossentropy')

callback = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, epsilon=1e-5, mode='max')
model.fit(X_train,Y_train,

          batch_size=128,

          validation_data=(X_val,Y_val),

          epochs=50,

         callbacks=[callback])
pred = model.predict(test_ohe)

pred
pred[1]
sub = pd.DataFrame(pred)

sub.head()
submission.head()
sub.insert(loc=0, column='sig_id', value=submission['sig_id'])

sub.columns = submission.columns

sub.head()
sub.to_csv('submission.csv',index=False)