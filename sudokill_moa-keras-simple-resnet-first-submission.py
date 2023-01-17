import numpy as np 

import pandas as pd 

import os 

import tensorflow as tf 

import keras

from tensorflow.keras import Input, Model 

from tensorflow.keras.models import Sequential, load_model

from tensorflow.keras.layers import Dense, Conv2D, Dropout, AlphaDropout, MaxPooling2D, AveragePooling2D, BatchNormalization, Concatenate, Flatten, Reshape, Add, Activation

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler

from keras.utils.np_utils import to_categorical

import matplotlib.pyplot as plt 

from sklearn.model_selection import train_test_split

from sklearn.utils import shuffle

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler 
train_ftr = pd.read_csv('../input/lish-moa/train_features.csv')

train_tgt = pd.read_csv('../input/lish-moa/train_targets_scored.csv') 

test_ftr = pd.read_csv('../input/lish-moa/test_features.csv') 

submission = pd.read_csv('../input/lish-moa/sample_submission.csv')



train_ftr.shape, train_tgt.shape, test_ftr.shape, submission.shape
# visualize train feature dataframe  

train_ftr.head()
ignore_columns = ['sig_id','cp_type'] 

train_columns = [x for x in train_ftr.columns if x not in ignore_columns]

train = train_ftr[train_columns] 

test = test_ftr[train_columns] 

target = train_tgt.iloc[:,1:].values 
le = LabelEncoder() 

train['cp_dose'] = le.fit_transform(train['cp_dose']) 

test['cp_dose'] = le.transform(test['cp_dose'])



train['cp_time'] = le.fit_transform(train['cp_time']) 

test['cp_time'] = le.transform(test['cp_time']) 
train = train.values 

test = test.values 
# column-wise standardization 

scalers = [] 

for i in range(2,train.shape[1]): 

    arr = train[:,i]

    arr = arr.reshape(-1,1) 

    sc = StandardScaler() 

    sc.fit(arr) 

    arr = sc.transform(arr) 

    arr = arr.reshape(arr.shape[0]) 

    train[:,i] = arr  

    scalers.append(sc)
for i in range(2, test.shape[1]): 

    sc = scalers[i-2] 

    arr = test[:,i] 

    arr = arr.reshape(-1,1)

    arr = sc.transform(arr) 

    arr = arr.reshape(arr.shape[0])

    test[:,i] = arr 
train.shape
def build_model(): 

    inputs = Input((874))

    dense1 = Dense(256, activation = 'relu')(inputs) 

    dense1 = BatchNormalization()(dense1) 

    dense2 = Dense(256, activation = 'relu')(dense1) 

    dense2 = BatchNormalization()(dense2) 

    dense3 = Dense(256, activation = 'relu')(dense2) 

    dense3 = Add()([dense1, dense3])  

    dense3 = BatchNormalization()(dense3) 

    outputs = Dropout(0.25)(dense3) 

    outputs = Dense(206, activation = 'sigmoid')(outputs)

    model = Model(inputs = inputs, outputs = outputs) 

    model.compile(optimizer = 'adam', loss = 'binary_crossentropy')

    return model 



model = build_model() 

model.summary()
k = int(0.9*len(train))

x_train = train[:k] 

y_train = target[:k] 



x_val = train[k:]

y_val = target[k:]




#checkpoint = ModelCheckpoint(filepath=model_path,monitor='val_loss',verbose=1,save_best_only=True)

learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', patience=3, verbose=1, factor=0.8)





history = model.fit(x_train,

                    y_train,

                    batch_size = 32,

                    shuffle = True, 

                    validation_data = (x_val,y_val),

                    verbose = 1, 

                    epochs = 5)

pred = model.predict(test)
for i in range(submission.shape[0]):

    submission.iloc[i,1:] = pred[i] 
submission
submission.to_csv('submission.csv', index = False)