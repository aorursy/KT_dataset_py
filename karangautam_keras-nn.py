import tensorflow as tf

import pandas as pd

import os

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from keras import Sequential

from keras import layers

from keras import backend as K

from keras.layers.core import Dense

from keras import regularizers

from keras.layers import Dropout

from keras.constraints import max_norm
import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

import lightgbm as lgb

from sklearn.model_selection import KFold

import warnings

import gc

import time

import sys

import datetime

import PIL, os, numpy as np, math, collections, threading, json,  random, scipy, cv2

import matplotlib.pyplot as plt

import seaborn as sns

from tqdm import tqdm

from sklearn.metrics import mean_squared_error

warnings.simplefilter(action='ignore', category=FutureWarning)

from sklearn import metrics

# Plotly library

from plotly.offline import init_notebook_mode, iplot

import plotly.graph_objs as go

from plotly import tools

init_notebook_mode(connected=True)

pd.set_option('display.max_columns', 500)

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import StratifiedKFold, KFold, TimeSeriesSplit

from sklearn.metrics import mean_squared_error, roc_auc_score

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

from sklearn import metrics

import gc

from catboost import CatBoostClassifier

from tqdm import tqdm_notebook

import plotly.offline as py
# Import data

train = pd.read_csv('../input/train.csv')

train.shape
#Check num of cases in label 

print(train.target.value_counts())

print(train.target.value_counts()[1]/train.target.value_counts()[0])
gc.collect()
train_features = train.drop(['target', 'ID_code'], axis=1)

train_targets = train['target']

train.describe()
train_features= pd.DataFrame(train_features)
from sklearn.preprocessing import power_transform

features = [c for c in train.columns if c not in ['ID_code', 'target']]

for feature in features:

    train_features['mean_'+feature] = (train_features[feature].mean()-train_features[feature])

    train_features['z_'+feature] = (train_features[feature] - train_features[feature].mean())/train_features[feature].std(ddof=0)

    train_features['sq_'+feature] = (train_features[feature])**2

    train_features['sqrt_'+feature] = (train_features['sq_'+feature])**(1/4)

    train_features['log_'+feature] = np.log(train_features['sq_'+feature]+10)/2
train_features.head()
# Feature Scaling

from sklearn.preprocessing import MinMaxScaler,StandardScaler

sc = StandardScaler()

train_features = sc.fit_transform(train_features)

gc.collect()
# Add RUC metric to monitor NN

def auc(y_true, y_pred):

    auc = tf.metrics.auc(y_true, y_pred)[1]

    K.get_session().run(tf.local_variables_initializer())

    return auc
input_dim = train_features.shape[1]

input_dim
from keras import callbacks

from sklearn.metrics import roc_auc_score



class printAUC(callbacks.Callback):

    def __init__(self, X_train, y_train):

        super(printAUC, self).__init__()

        self.bestAUC = 0

        self.X_train = X_train

        self.y_train = y_train

        

    def on_epoch_end(self, epoch, logs={}):

        pred = self.model.predict(np.array(self.X_train))

        auc = roc_auc_score(self.y_train, pred)

        print("Train AUC: " + str(auc))

        #pred = self.model.predict(self.validation_data[0])

        #auc = roc_auc_score(self.validation_data[1], pred)

        #print ("Validation AUC: " + str(auc))

        if (self.bestAUC < auc) :

            self.bestAUC = auc

            self.model.save("bestNet.h5", overwrite=True)

        return
from keras.layers import Dense,Dropout,BatchNormalization

from keras import regularizers

import keras

from keras.callbacks import LearningRateScheduler,EarlyStopping

import tensorflow as tf

from keras import backend as K

from sklearn.model_selection import train_test_split

from keras.constraints import max_norm
def step_decay(epoch):

   initial_lrate = 0.1

   drop = 0.5

   epochs_drop = 10.0

   lrate = initial_lrate * math.pow(drop,  

           math.floor((1+epoch)/epochs_drop))

   return lrate

lrate = LearningRateScheduler(step_decay)
class LossHistory(keras.callbacks.Callback):

    def on_train_begin(self, logs={}):

       self.losses = []

       self.lr = []

 

    def on_epoch_end(self, batch, logs={}):

       self.losses.append(logs.get('loss'))

       self.lr.append(step_decay(len(self.losses)))
import random

from keras import models

from keras import regularizers

from keras.callbacks import LearningRateScheduler

from keras.layers.advanced_activations import PReLU,LeakyReLU

kernel_regularizer=regularizers.l2(0.01)

model = models.Sequential()

model.add(Dense(128, activation='relu', input_shape=(train_features.shape[1],)))

#model.add(PreLU(alpha=.001))

model.add(Dropout(0.6))

model.add(BatchNormalization())

model.add(Dense(64,activation='relu'))

#model.add(PreLU(alpha=.001))

model.add(Dropout(0.6))

model.add(BatchNormalization())

model.add(Dense(32,activation='relu'))

#model.add(PreLU(alpha=.001))

model.add(Dropout(0.5))

model.add(BatchNormalization())

model.add(Dense(1,activation='sigmoid'))



annealer = LearningRateScheduler(lambda x: 1e-2 * 0.95 ** x)
model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy',auc])

gc.collect()
from sklearn.model_selection import StratifiedShuffleSplit

loss_history = LossHistory()

lrate = LearningRateScheduler(step_decay)

callbacks_list = [EarlyStopping(monitor='val_auc', patience=10,mode='max'),loss_history, annealer]

sss = StratifiedShuffleSplit(n_splits=10)

for train_index, test_index in sss.split(train_features, train_targets):

    X_train, X_val = train_features[train_index], train_features[test_index]

    Y_train, Y_val = train_targets[train_index], train_targets[test_index]

#    print("{} iteration".format(i+1))

    history= model.fit(X_train,Y_train,batch_size=512,epochs=50,callbacks=callbacks_list,verbose=1,validation_data=(X_val,Y_val))

    del X_train, X_val, Y_train, Y_val

    gc.collect()
# Try early stopping

#from keras.callbacks import EarlyStopping

#callback = EarlyStopping(monitor='loss', min_delta=0, patience=10, verbose=0, mode='auto', baseline=None, restore_best_weights=True)
train_features.shape
del train, train_features

gc.collect()
test = pd.read_csv('../input/test.csv')

test_features = test.drop(['ID_code'], axis=1)
for feature in features:

    test_features['mean_'+feature] = (test_features[feature].mean()-test_features[feature])

    test_features['z_'+feature] = (test_features[feature] - test_features[feature].mean())/test_features[feature].std(ddof=0)

    test_features['sq_'+feature] = (test_features[feature])**2

    test_features['sqrt_'+feature] = (test_features['sq_'+feature])**(1/4)

    test_features['log_'+feature] = np.log(test_features['sq_'+feature]+10)/2
test_features = sc.transform(test_features)
id_code_test = test['ID_code']

# Make predicitions

pred = model.predict(test_features)

pred_ = pred[:,0]
print(train['target'].mean())

pred.mean()
# To CSV

my_submission = pd.DataFrame({"ID_code" : id_code_test, "target" : pred_})
my_submission.to_csv('submission.csv', index = False, header = True)