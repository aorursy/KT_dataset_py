import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os 

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns # data visualization library  

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import time

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from keras.preprocessing.text import *

from keras.preprocessing import sequence



from sklearn.feature_extraction.text import TfidfVectorizer

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

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import KFold, RepeatedKFold



import os 

print(os.listdir('../input/'))

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

print(os.listdir('../input/music-hit-prediction'))
train_df = pd.read_csv('../input/music-hit-prediction/train_info.tsv',delimiter='\t',encoding='utf-8')

test_df = pd.read_csv('../input/music-hit-prediction/test_info.tsv',delimiter='\t',encoding='utf-8')

train_label = pd.read_csv('../input/music-hit-prediction/train_rank.csv')
train_df.head()
train_label.head()
print(train_df.shape)

print(train_label.shape)
all_data = pd.concat((train_df.loc[:,'ID':'release_time'],

                      test_df.loc[:,'ID':'release_time']))

all_data.head()
from sklearn import preprocessing

le = preprocessing.LabelEncoder()



le.fit(all_data['artist_id']) 

all_data['artist_id'] = le.transform(all_data['artist_id'])



le.fit(all_data['composers_id']) 

all_data['composers_id'] = le.transform(all_data['composers_id'])
from tqdm import tqdm

# Preprocess text

print('Preprocessing text...')

cols = [

    'artist_name', 

    'composers_name', 

    'title'

]

n_features = [

    60, 

    60, 

    60,

]



for c_i, c in tqdm(enumerate(cols)):

    tfidf = TfidfVectorizer(

        max_features=n_features[c_i],

        norm='l1',

        )

    tfidf.fit(all_data[c])

    tfidf_train = np.array(tfidf.transform(train_df[c]).toarray(), dtype=np.float16)

    tfidf_test = np.array(tfidf.transform(test_df[c]).toarray(), dtype=np.float16)



    for i in range(n_features[c_i]):

        train_df[c + '_tfidf_' + str(i)] = tfidf_train[:, i]

        test_df[c + '_tfidf_' + str(i)] = tfidf_test[:, i]

        

    del tfidf, tfidf_train, tfidf_test

#     gc.collect(
all_data = pd.concat((train_df.loc[:,'title':'title_tfidf_59'],

                      test_df.loc[:,'title':'title_tfidf_59']))
from sklearn import preprocessing

le = preprocessing.LabelEncoder()



le.fit(all_data['artist_id']) 

all_data['artist_id'] = le.transform(all_data['artist_id'])



le.fit(all_data['composers_id']) 

all_data['composers_id'] = le.transform(all_data['composers_id'])



# all_data = all_data.drop(['title', 'artist_name', 'composers_name'], axis = 1)



all_data.head()
all_data['release_time'] = pd.to_datetime(all_data['release_time'], errors='coerce')



all_data['hour'] = all_data['release_time'].dt.hour

all_data['dayofweek'] = all_data['release_time'].dt.dayofweek

all_data['quarter'] = all_data['release_time'].dt.quarter

all_data['month'] = all_data['release_time'].dt.month

all_data['year'] = all_data['release_time'].dt.year



# all_data['dayofyear'] = all_data['release_time'].dt.dayofyear

# all_data['dayofmonth'] = all_data['release_time'].dt.day

# all_data['weekofyear'] = all_data['release_time'].dt.weekofyear



all_data.head()
all_data = all_data.drop(['title', 'artist_name', 'composers_name'], axis = 1)
all_data = all_data.drop(['release_time'], axis = 1)

print(all_data.shape)

all_data.head()
train_label.head()
train_features = all_data[:train_df.shape[0]]

train_targets = train_label['label']

test_features = all_data[train_df.shape[0]:]
from numpy import array

from keras.models import Sequential

from keras.layers import Dense

from matplotlib import pyplot

from keras import backend

 

def rmse(y_true, y_pred):

    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))
input_dim = train_features.shape[1]

input_dim
# Model

model = Sequential()

model.add(Dense(256, input_dim=input_dim, kernel_initializer='normal', activation='relu'))

model.add(Dropout(rate=0.2))

# model.add(Dense(512, kernel_initializer='normal', activation='relu'))

# model.add(Dropout(rate=0.3))

# model.add(Dense(256, kernel_initializer='normal', activation='relu'))

# model.add(Dropout(rate=0.2))

model.add(Dense(128, kernel_initializer='normal', activation='relu'))

model.add(Dropout(rate=0.3))

model.add(Dense(64, kernel_initializer='normal', activation='relu'))

model.add(Dropout(rate=0.2))

model.add(Dense(32, kernel_initializer='normal', activation='relu'))

model.add(Dropout(rate=0.3))

model.add(Dense(16, kernel_initializer='normal', activation='relu'))

model.add(Dropout(rate=0.2))

model.add(Dense(1, kernel_initializer='normal', activation='relu'))

# Compile model

model.compile(loss=[rmse], optimizer='adam', metrics=[rmse])

model.summary()
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau



checkpoint = ModelCheckpoint('feed_forward_model.h5', monitor='val_loss', verbose=1, 

                             save_best_only=True, mode='min', save_weights_only = True)

reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, 

                                   verbose=1, mode='min', epsilon=0.0001)

early = EarlyStopping(monitor="val_loss", 

                      mode="min", 

                      patience=10)

callbacks_list = [early, checkpoint, reduceLROnPlat]
test_features.shape
train_targets = pd.DataFrame(train_targets)
train_targets.head()
from sklearn.metrics import mean_squared_error

import math  

from sklearn.model_selection import KFold, StratifiedKFold



# import lightgbm as lgb



score = []



skf = StratifiedKFold(n_splits = 5, shuffle=True, random_state=123)

skf.get_n_splits(train_features, train_targets)



features = [col for col in train_features.columns]

feature_importance_df = pd.DataFrame()

predictions = pd.DataFrame(test_df['ID'])

x_test = test_features



for fold, (trn_idx, val_idx) in enumerate(skf.split(train_features, train_targets)):

#     print("FOLD: ", fold, "TRAIN:", trn_idx, "VALID:", val_idx)

    x_train, y_train = train_features.iloc[trn_idx][features], train_targets.iloc[trn_idx]['label']

    x_valid, y_valid = train_features.iloc[val_idx][features], train_targets.iloc[val_idx]['label']



    p_valid = 0

    yp = 0

#     for i in range(N):

    model.compile(loss=[rmse], optimizer='adam', metrics=[rmse])

    model.fit(x_train, y_train, batch_size = 1024, epochs = 125,

          validation_data = (x_valid, y_valid),

          callbacks = callbacks_list)

    model.load_weights('feed_forward_model.h5')

    score.append(math.sqrt(mean_squared_error(model.predict(x_valid), y_valid)))

    yp += model.predict(x_test)       

    fold_importance_df = pd.DataFrame()

    fold_importance_df["feature"] = features

#     fold_importance_df["importance"] = gbm.feature_importance()

    fold_importance_df["fold"] = fold + 1

    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    predictions['fold{}'.format(fold+1)] = yp
score = pd.DataFrame(score)

print(score[0].mean())

print(score[0].std())

print(score[0])