import os

import pandas as pd

import numpy as np

import zipfile

import tensorflow as tf

from tensorflow.keras.optimizers import RMSprop,Adam

import tensorflow_addons as tfa

from tensorflow import keras

from shutil import copyfile

from os import getcwd

import matplotlib.pyplot as plt

from sklearn.model_selection import KFold

from sklearn import preprocessing

from sklearn.metrics import confusion_matrix

from sklearn.feature_selection import VarianceThreshold

from sklearn.decomposition import PCA

from tensorflow.keras import layers,regularizers,Sequential,Model,backend,callbacks,optimizers,metrics,losses

import tensorflow as tf

import sys

import json

sys.path.append('../input/iterative-stratification/iterative-stratification-master')

#from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


df = pd.read_csv('/kaggle/input/lish-moa/train_features.csv')

non_ctl_idx = df.loc[df['cp_type']!='ctl_vehicle'].index.to_list()

df.head()
df['cp_type'] = pd.Categorical(df['cp_type'])

df['cp_type'] = df.cp_type.cat.codes

df['cp_dose'] = pd.Categorical(df['cp_dose'])

df['cp_dose'] = df.cp_dose.cat.codes

df['cp_time'] = pd.Categorical(df['cp_time'])

df['cp_time'] = df.cp_time.cat.codes

dada = pd.read_csv('/kaggle/input/lish-moa/train_targets_scored.csv')

df = df.iloc[non_ctl_idx]

dada = dada.iloc[non_ctl_idx]

#train = pd.concat([df,dada],axis=1,sort=False)

#train = train.sample(frac = 1) 

df.head(5)
#train_xy = train.iloc[:19051,:]

#validation_xy = train.iloc[19051:,:]
#train_x = train_xy.iloc[:,1:876]

#train_y = train_xy.iloc[:,877:]

#train_x['cp_type'] = train_x['cp_type'].astype(float)

#train_x['cp_time'] = train_x['cp_time'].astype(float)

#train_x['cp_dose'] = train_x['cp_dose'].astype(float)

#train_x = np.array(train_x)

#train_y = np.array(train_y)

#validation_x = validation_xy.iloc[:,1:876]

#validation_y = validation_xy.iloc[:,877:]

#validation_x['cp_type'] = validation_x['cp_type'].astype(float)

#validation_x['cp_time'] = validation_x['cp_time'].astype(float)

#validation_x['cp_dose'] = validation_x['cp_dose'].astype(float)

#validation_x = np.array(validation_x)

#validation_y = np.array(validation_y)


p_min = 0.0005

p_max = 0.9995



def logloss(y_true, y_pred):

    y_pred = tf.clip_by_value(y_pred,p_min,p_max)

    return -backend.mean(y_true*backend.log(y_pred) + (1-y_true)*backend.log(1-y_pred))
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_logloss', factor=0.1, verbose=1,mode='min',

                              patience=2, min_lr=0.0000001)

early_st = tf.keras.callbacks.EarlyStopping(monitor='val_logloss', min_delta=1E-5, patience=4, verbose=1, mode='min',

    baseline=None, restore_best_weights=True)
model = tf.keras.Sequential([

  tf.keras.layers.Input(875),

  tf.keras.layers.BatchNormalization(),

  tf.keras.layers.Dropout(0.3),

  #tfa.layers.WeightNormalization(tf.keras.layers.Dense(1024,activation='relu')),

  #tf.keras.layers.BatchNormalization(),

  #tf.keras.layers.Dropout(0.55),

  tfa.layers.WeightNormalization(tf.keras.layers.Dense(1024,activation='relu')),

  tf.keras.layers.BatchNormalization(),

  tf.keras.layers.Dropout(0.5),

  tfa.layers.WeightNormalization(tf.keras.layers.Dense(512,activation='relu')),

  tf.keras.layers.BatchNormalization(),

  tf.keras.layers.Dropout(0.3),

  tfa.layers.WeightNormalization(tf.keras.layers.Dense(206,activation='sigmoid'))

])







model.compile(optimizer=tfa.optimizers.Lookahead(tf.optimizers.Adam()),

                  loss='binary_crossentropy', metrics=logloss 

                  )
from keras import backend 

n_seeds = 7

np.random.seed(1)

seeds = np.random.randint(0,100,size=n_seeds)



# Training Loop

n_labels = dada.shape[1]

n_train = df.shape[0]

n_test = df.shape[0]

n_folds = 10

y_pred = np.zeros((n_test,n_labels))

oof = tf.constant(0.0)

hists = []

for seed in seeds:

    fold = 0

    kf = KFold(n_splits=n_folds,shuffle=True,random_state=seed)

    for train, test in kf.split(df):

        train_x = df.iloc[train]

        train_x = train_x.iloc[:,1:876]

        train_x['cp_type'] = train_x['cp_type'].astype(float)

        train_x['cp_time'] = train_x['cp_time'].astype(float)

        train_x['cp_dose'] = train_x['cp_dose'].astype(float)

        train_y = dada.iloc[train]

        train_y = train_y.iloc[:,1:207]

        validation_x = df.iloc[test]

        validation_x = validation_x.iloc[:,1:876]

        validation_x['cp_type'] = validation_x['cp_type'].astype(float)

        validation_x['cp_time'] = validation_x['cp_time'].astype(float)

        validation_x['cp_dose'] = validation_x['cp_dose'].astype(float)

        validation_y = dada.iloc[test]

        validation_y = validation_y.iloc[:,1:207]

        

        train_x = np.array(train_x)

        train_y = np.array(train_y)

        validation_x = np.array(validation_x)

        validation_y = np.array(validation_y)

        



        hist = model.fit(train_x,train_y, batch_size=128, epochs=192,verbose=2,validation_data = (validation_x,validation_y),callbacks =[reduce_lr, early_st])

        hists.append(hist)

        

        # Save Model

        model.save('TwoHeads_seed_'+str(seed)+'_fold_'+str(fold))



        # OOF Score

        y_val = model.predict(validation_x)

        oof += logloss(tf.constant(validation_y,dtype=tf.float32),tf.constant(y_val,dtype=tf.float32))/(n_folds*n_seeds)



        # Run prediction

        y_pred += model.predict(train_x)/(n_folds*n_seeds)



        fold += 1
#from keras import backend 

#print("Fit model on training data")

#history = model.fit

    #train_x,

    #train_y,

    #epochs=50,

    #validation_data=(validation_x,validation_y),

    #callbacks = [reduce_lr,early_st]

#
test = pd.read_csv('/kaggle/input/lish-moa/test_features.csv')

test['cp_type'] = pd.Categorical(test['cp_type'])

test['cp_type'] = test.cp_type.cat.codes

test['cp_dose'] = pd.Categorical(test['cp_dose'])

test['cp_dose'] = test.cp_dose.cat.codes

test['cp_time'] = pd.Categorical(test['cp_time'])

test['cp_time'] = test.cp_time.cat.codes

test_x = test.iloc[:,1:]

test_con = test.iloc[:,0]
test_y = model.predict(test_x)

train_yy = pd.read_csv('/kaggle/input/lish-moa/train_targets_scored.csv')

aa = list(train_yy.columns[1:])

data = pd.DataFrame(test_y,columns=aa,index=test['sig_id'])

data.to_csv('submission.csv')