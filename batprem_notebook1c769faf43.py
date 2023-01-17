# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os


# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
seed_value= 0

import os
os.environ['PYTHONHASHSEED']=str(seed_value)

import random
random.seed(seed_value)

import numpy as np
np.random.seed(seed_value)

import tensorflow as tf
tf.random.set_seed(seed_value)
train_features = pd.read_csv('../input/lish-moa/train_features.csv')
X_train = train_features.drop('sig_id', axis = 1)
train_targets_scored = pd.read_csv('../input/lish-moa/train_targets_scored.csv')
Y_train = train_targets_scored.drop('sig_id', axis = 1)
combine = pd.concat([X_train, Y_train], axis=1, sort=False)
y_col = Y_train.columns[1]
combine.loc[Y_train[Y_train[y_col]==1].index]
all_zero = True
for yc in Y_train.columns:
    temp = (Y_train[yc] == 0)
    if type(all_zero) is bool :
        all_zero = temp
    else:
        all_zero = all_zero & temp
X_train_wraggled = X_train[X_train['cp_type'] != 'ctl_vehicle'][X_train.columns[3:]]
Y_train_wraggled = Y_train.loc[X_train_wraggled.index]
X_train_wraggled_g = X_train_wraggled[X_train_wraggled.columns[pd.Series(X_train_wraggled.columns).str.startswith('g')]] 

X_train_wraggled_c = X_train_wraggled[X_train_wraggled.columns[pd.Series(X_train_wraggled.columns).str.startswith('c')]] 
print(X_train_wraggled.shape[1])
print(X_train_wraggled_g.shape[1])
print(X_train_wraggled_c.shape[1])
# Fully connect
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras.layers import Input, Concatenate, concatenate, BatchNormalization
from keras.models import Model
from keras.layers import Dense, Conv1D, GlobalMaxPooling1D
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.callbacks import TensorBoard
from tensorflow_addons.layers import WeightNormalization
from tensorflow.keras import regularizers
from tensorflow.keras.losses import BinaryCrossentropy
from keras.models import load_model
import keras

from keras import backend as K

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
              

model_name = "conv"
model_name = "parallel_conv"
# model_name = 'g_conv'
# model_name = 'c_lstm'
#Load pretrained model
from keras.models import load_model

import tensorflow as tf

def f1(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)

def f1_loss(y_true, y_pred):
    
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return 1 - K.mean(f1)


class_weight = {0: 0.01,
                1: 1}
#Create model function
def getModel(model_name):
    if model_name == 'pretrained':
        model = load_model('../input/moa-lstm/best_weights (1).hdf5')
        toBeCompiled = False
        
    elif model_name == 'parallel_conv':
        InputLayer_g = Input(shape=(X_train_wraggled_g.shape[1], 1))
        InputLayer_c = Input(shape=(X_train_wraggled_c.shape[1], 1))
        
        
        ConvLayer_g = Conv1D(filters=200,
                           kernel_size=20,
                           padding='valid',
                           activation='relu',
                           strides=1)(InputLayer_g)
        PoolingLayer_g = BatchNormalization()(ConvLayer_g)
        PoolingLayer_g = GlobalMaxPooling1D()(ConvLayer_g)
        PoolingLayer_g = BatchNormalization()(PoolingLayer_g)
        PoolingLayer_g = Dropout(0.4)(PoolingLayer_g)
        
        ConvLayer_c = Conv1D(filters=200,
                           kernel_size=20,
                           padding='valid',
                           activation='relu',
                           strides=1)(InputLayer_c)
        PoolingLayer_c = BatchNormalization()(ConvLayer_c)
        PoolingLayer_c = GlobalMaxPooling1D()(ConvLayer_c)
        PoolingLayer_c = BatchNormalization()(PoolingLayer_c)
        PoolingLayer_c = Dropout(0.4)(PoolingLayer_c)
        
        merged = concatenate([PoolingLayer_g, PoolingLayer_c], axis=1)
        merged = BatchNormalization()(merged)
        merged = Dropout(0.2)(merged)
        merged = WeightNormalization(Dense(300, activation='relu'))(merged)
        merged = BatchNormalization()(merged)
        merged = Dropout(0.4)(merged)
        #OutputLayer = WeightNormalization(Dense(206, activation='sigmoid'))(merged)
        OutputLayer = WeightNormalization(Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.001)))(merged)
        model = Model(inputs=[InputLayer_g,InputLayer_c], outputs=OutputLayer)
        toBeCompiled = True
        
    return toBeCompiled,model

def fit1Label(col):
    #class_weight = calculating_class_weights(Y_train_wraggled[[col]].iloc[0:int(len(X_train_wraggled)*0.75)].values)
    class_weight = {
        0:1,
        1:1.2
    }
    bi_loss = BinaryCrossentropy(
        from_logits=False,
        label_smoothing=0, 
        reduction="auto", 
        name="binary_crossentropy")
    METRICS = [
        'accuracy',
        "binary_crossentropy",
        f1_m,
        keras.metrics.TruePositives(name='tp'),
        keras.metrics.FalsePositives(name='fp'),
        keras.metrics.TrueNegatives(name='tn'),
        keras.metrics.FalseNegatives(name='fn'), 
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall'),
        keras.metrics.AUC(name='auc'),
    ]

    #Set Checkpoint
    filepath=f"{col}.hdf5"
    if filepath in os.listdir():
        print(f"skip {filepath}")
        return False
    checkpoint = ModelCheckpoint(filepath, monitor="val_binary_crossentropy", verbose=1, save_best_only=True, mode='min')
    # checkpoint = ModelCheckpoint(filepath, monitor="val_tp", verbose=1, save_best_only=True, mode='max')
    early_stop =EarlyStopping(monitor="val_tp", mode = 'min', patience=15)
    toBeCompiled,model = getModel(model_name)

    model.compile(loss=bi_loss,
                  optimizer='adam',
                  metrics=METRICS)
    
    
    model.fit([
                X_train_wraggled_g.iloc[0:int(len(X_train_wraggled_g)*0.75)].values,
                X_train_wraggled_c.iloc[0:int(len(X_train_wraggled_c)*0.75)].values
                ],
              Y_train_wraggled[[col]].iloc[0:int(len(X_train_wraggled)*0.75)].values.astype('float'),
              epochs=600,
              validation_data=(
                      [
                        X_train_wraggled_g.iloc[int(len(X_train_wraggled_g)*0.75):].values,
                        X_train_wraggled_c.iloc[int(len(X_train_wraggled_c)*0.75):].values
                    ],
                      Y_train_wraggled[[col]].iloc[int(len(X_train_wraggled)*0.75):].values
                  ),
              #batch_size=1000,
                shuffle=True,
        #([[ 0.50441258, 57.15625   ]])
                class_weight=class_weight,
              callbacks = [checkpoint,early_stop]
             )
    del model
    
def evaluate_models(col):
    model = load_model(f'../input/pretrainmodel/{col}.hdf5', custom_objects={'f1_m':f1_m})
    res = model.evaluate([
                            X_train_wraggled_g.iloc[int(len(X_train_wraggled)*0.75):],
                            X_train_wraggled_c.iloc[int(len(X_train_wraggled)*0.75):]
                        ], 
                         Y_train_wraggled.iloc[int(len(X_train_wraggled)*0.75):][[col]])
    print(f'Model {col}: Binary cross: {res[2]}')
    return res

#import data

def col_prediction(col):
    model = load_model(f'../input/pretrainmodel/{col}.hdf5', custom_objects={'f1_m':f1_m})
    res = model([
                    X_test_wraggled_g.values,
                    X_test_wraggled_c.values
                ])
    return np.array(res).flatten()

test_features = pd.read_csv('../input/lish-moa/test_features.csv')

X_test = test_features

X_test_wraggled = X_test[X_test.columns[4:]]

X_test_wraggled_g = X_test_wraggled[X_test_wraggled.columns[pd.Series(X_test_wraggled.columns).str.startswith('g')]] 

X_test_wraggled_c = X_test_wraggled[X_test_wraggled.columns[pd.Series(X_test_wraggled.columns).str.startswith('c')]] 

predicts = X_test[['sig_id']]

for count,col in enumerate(Y_train_wraggled):
    print('='*36)
    print(col)
    predict = col_prediction(col)
    predicts[col] = predict
predicts.to_csv('submission.csv', index=False)
predicts.shape