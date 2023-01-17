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



import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib

from sklearn import preprocessing 

import matplotlib.pyplot as plt

from scipy.stats import skew

from scipy.stats.stats import pearsonr





%config InlineBackend.figure_format = 'retina' #set 'png' here when working on notebook

%matplotlib inline



from sklearn.model_selection import cross_val_score

from sklearn.model_selection import train_test_split

import xgboost

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split, StratifiedKFold

from sklearn.preprocessing import MinMaxScaler, StandardScaler, scale

from sklearn.metrics import roc_auc_score



from keras.layers import Activation, Dropout, Flatten, Dense, GlobalMaxPooling2D, BatchNormalization, Input, Conv2D

from keras.callbacks import ModelCheckpoint

from keras import metrics

from keras.optimizers import Adam 

from keras import backend as K

import keras

from keras.models import Model

from keras import Sequential

from keras import regularizers

import tensorflow as tf

from keras.losses import binary_crossentropy

import gc

import scipy.special

from tqdm import *

from scipy.stats import norm, rankdata



from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
train = pd.read_csv('../input/dataset/Train_full.csv')

test = pd.read_csv('../input/dataset-v2/Test_small_features.csv')
arr = []

for i in range(test.shape[0]):

    if i == 0:

        continue

    else:

        if test.at[i, 'body'] > 0:

            arr.append(1)

        else:

            arr.append(0)

arr.append(0)            

y_true = pd.DataFrame(arr)
print(train.shape)

print(test.shape)
train.head()
test.head()
all_data = pd.concat((train.loc[:,'Open':'lag_return_96'],

                      test.loc[:,'Open':'lag_return_96']))

all_data.head()
all_data = all_data.drop(['Volume', 'upper_tail','lower_tail'], axis = 1)
cat_feat = ['hour', 'min', 'dayofweek']
cat_data = all_data[cat_feat]
all_data = all_data.drop(cat_feat, axis = 1)

all_data.head()
numeric_feats = all_data.columns

skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness

# print(skewed_feats)

skewed_feats = skewed_feats[skewed_feats > 0.75]

# print(skewed_feats)

# all_data = all_data.

skewed_feats = skewed_feats.index

all_data[skewed_feats] = np.log1p(all_data[skewed_feats])

all_data.head()
all_data = pd.concat([all_data,cat_data], axis = 1)

all_data.head()
X_train = all_data[:train.shape[0]]

X_test = all_data[train.shape[0]:]

y = train.up_down
# dayofweek = X_train['dayofweek']

x_train, x_valid, y_train, y_valid = train_test_split(X_train, y, test_size = 0.2, random_state = 8, shuffle = False)
import lightgbm as lgb
train_data = lgb.Dataset(x_train, y_train, free_raw_data=False, categorical_feature = cat_feat)

valid_data = lgb.Dataset(x_valid, y_valid, free_raw_data=False, categorical_feature = cat_feat)
params = {

    'task': 'train',

    'boosting_type': 'gbdt',

    'objective': 'binary',

    'metric': 'binary_logloss',

    'num_leaves': 96,

    'max_depth': 10,

    'feature_fraction': 0.9,

    'bagging_fraction': 0.95,

    'bagging_freq': 5

}

from lightgbm import LGBMClassifier

num_round = 1000

lgbm = LGBMClassifier(num_leaves= 180, max_depth= -1, n_estimators = 2000, n_jobs = 16, random_state = 4, subsample = 0.9, gpu_id = 0, colsample_bytree = 0.85, max_bin = 512, tree_method = 'gpu_hist')

lgbm.fit(X=x_train,y=y_train,eval_set = [(x_train,y_train),(x_valid, y_valid)], eval_metric = ['binary_logloss'], early_stopping_rounds = 70)

# model = lgb.train(parameter, train_data, num_round, valid_sets = [train_data, valid_data], verbose_eval = 100, early_stopping_rounds = 50)
from xgboost import XGBClassifier

model = XGBClassifier(max_depth = 5, n_estimators = 2000, n_jobs = 16, random_state = 4, subsample = 0.9, gpu_id = 0, colsample_bytree = 0.80, max_bin = 16, tree_method = 'gpu_hist')

model.fit(X=x_train,y=y_train,eval_set = [(x_train,y_train),(x_valid, y_valid)], eval_metric = ['logloss'], early_stopping_rounds = 70)
pred_lgb = lgbm.predict_proba(X_test)
pred_xg = model.predict_proba(X_test)
lgbpred= pd.DataFrame(pred_lgb)
xgpred= pd.DataFrame(pred_xg)
all_data = pd.concat((train.loc[:,'Open':'lag_return_96'],

                      test.loc[:,'Open':'lag_return_96']))

all_data.head()
hour = pd.get_dummies(all_data['hour'])

min_data = pd.get_dummies(all_data['min'])

day_data = pd.get_dummies(all_data['dayofweek'])
all_data = all_data.drop(cat_feat,axis = 1)
for i in hour.columns:

    hour.rename(columns={i:'hour_%s'%i}, inplace=True)
for i in min_data.columns:

    min_data.rename(columns={i:'min_%s'%i}, inplace=True)
for i in day_data.columns:

    day_data.rename(columns={i:'day_%s'%i}, inplace=True)
all_data = pd.concat([all_data, hour, min_data, day_data], axis=1)
features = all_data.columns
# Feature Scaling

sc = MinMaxScaler()

# x_train = sc.fit_transform(x_train)

# x_test = sc.transform(x_test)

# test_features = sc.transform(test_features)

scaled_df = sc.fit_transform(all_data)

all_data = pd.DataFrame(scaled_df, columns = features)
train_features = all_data[:train.shape[0]]

train_targets = train['up_down']

test_features = all_data[train.shape[0]:]
x_train, x_test, y_train, y_test = train_test_split(train_features, train_targets,

                                                    test_size = 0.2, random_state = 50, shuffle = False)
def auc(y_true, y_pred):

    auc = tf.metrics.auc(y_true, y_pred)[1]

    K.get_session().run(tf.local_variables_initializer())

    return auc
input_dim = x_train.shape[1]

input_dim
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
model = Sequential()

# Input layer

model.add(Dense(units = 512, activation = "relu", input_dim = input_dim,

                kernel_initializer = "he_normal",

                kernel_regularizer=regularizers.l2(5e-4)))

# Add dropout regularization

model.add(Dropout(rate=0.2))



# Second hidden layer

model.add(Dense(256, activation='relu',

                kernel_regularizer=regularizers.l2(5e-4)))

# Add dropout regularization

model.add(Dropout(rate=0.3))



# Third hidden layer

model.add(Dense(128, activation='relu',

                kernel_regularizer=regularizers.l2(5e-4)))

# Add dropout regularization

model.add(Dropout(rate=0.2))



# Output layer

model.add(layers.Dense(units = 1, activation='sigmoid'))



model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', auc])

model.summary()
checkpoint = ModelCheckpoint('feed_forward_model.h5', monitor='val_loss', verbose=1, 

                             save_best_only=True, mode='min', save_weights_only = True)

reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, 

                                   verbose=1, mode='min', epsilon=0.0001)

early = EarlyStopping(monitor="val_loss", 

                      mode="min", 

                      patience=10)
callbacks_list = [early, checkpoint, reduceLROnPlat]
model.fit(x_train, y_train, batch_size = 2048, epochs = 125,

          validation_data = (x_test, y_test),

          callbacks = callbacks_list)
model.load_weights('feed_forward_model.h5')

prediction = model.predict(test_features, batch_size=512, verbose=1)
pred_nn = pd.DataFrame(prediction)
pred_nn
pred_nn[1] = pred_nn[0]
pred = 0.7*pred_lgb + 0.2*pred_xg + 0.1*pred_nn
pred.head()
res = []

for i in range(pred.shape[0]):

    if pred.at[i, 1] > 0.505:

        res.append(1)

    else:

        res.append(0)

y_pred = pd.DataFrame(res)
from sklearn.metrics import accuracy_score

accuracy_score(y_pred, y_true)
mysubmit = pd.DataFrame({'up_down': res})
mysubmit.to_csv('submission.csv', index=True)