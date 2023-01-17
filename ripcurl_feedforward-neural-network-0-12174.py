# Thanks to the1owl for the data reading code, https://www.kaggle.com/the1owl/regressing-during-insomnia-0-21496
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
from sklearn.preprocessing import StandardScaler,MinMaxScaler

from sklearn.svm import OneClassSVM

import collections



# For plotting

%matplotlib inline

import matplotlib.pyplot as plt



from sklearn.metrics import (confusion_matrix, precision_recall_curve, auc,

                             roc_curve, recall_score, classification_report, f1_score,

                             precision_recall_fscore_support)
train = pd.read_csv('../input/train.csv')

train = pd.concat((train, pd.read_csv('../input/train_v2.csv')),axis=0, ignore_index=True).reset_index(drop=True)

test = pd.read_csv('../input/sample_submission_v2.csv')
transactions = pd.read_csv('../input/transactions.csv', usecols=['msno'])

transactions = pd.concat((transactions, pd.read_csv('../input/transactions_v2.csv', usecols=['msno'])), axis=0, ignore_index=True).reset_index(drop=True)

transactions = pd.DataFrame(transactions['msno'].value_counts().reset_index())

transactions.columns = ['msno','trans_count']

train = pd.merge(train, transactions, how='left', on='msno')

test = pd.merge(test, transactions, how='left', on='msno')
transactions = pd.read_csv('../input/transactions_v2.csv') 

transactions = transactions.sort_values(by=['transaction_date'], ascending=[False]).reset_index(drop=True)

transactions = transactions.drop_duplicates(subset=['msno'], keep='first')



train = pd.merge(train, transactions, how='left', on='msno')

test = pd.merge(test, transactions, how='left', on='msno')

transactions=[]
user_logs = pd.read_csv('../input/user_logs_v2.csv', usecols=['msno'])

user_logs = pd.DataFrame(user_logs['msno'].value_counts().reset_index())

user_logs.columns = ['msno','logs_count']

train = pd.merge(train, user_logs, how='left', on='msno')

test = pd.merge(test, user_logs, how='left', on='msno')



user_logs = []; 
def transform_df(df):

    df = pd.DataFrame(df)

    df = df.sort_values(by=['date'], ascending=[False])

    df = df.reset_index(drop=True)

    df = df.drop_duplicates(subset=['msno'], keep='first')

    return df



def transform_df2(df):

    df = df.sort_values(by=['date'], ascending=[False])

    df = df.reset_index(drop=True)

    df = df.drop_duplicates(subset=['msno'], keep='first')

    return df
last_user_logs = []

last_user_logs.append(transform_df(pd.read_csv('../input/user_logs_v2.csv')))

last_user_logs = pd.concat(last_user_logs, axis=0, ignore_index=True).reset_index(drop=True)

last_user_logs = transform_df2(last_user_logs)

print ('merging user logs features...')

train = pd.merge(train, last_user_logs, how='left', on='msno')

test = pd.merge(test, last_user_logs, how='left', on='msno')

last_user_logs=[]
members = pd.read_csv('../input/members_v3.csv')

train = pd.merge(train, members, how='left', on='msno')

test = pd.merge(test, members, how='left', on='msno')

members = []; print('members merge...') 
gender = {'male':1, 'female':2}

train['gender'] = train['gender'].map(gender)

test['gender'] = test['gender'].map(gender)



train = train.fillna(0)

test = test.fillna(0)
train = train.fillna(0)

test = test.fillna(0)

train.head()
# For Keras

from keras.models import Sequential

from keras.callbacks import ReduceLROnPlateau

from keras.callbacks import LambdaCallback

from keras.layers.normalization import BatchNormalization

from keras.optimizers import SGD

from keras.constraints import maxnorm

from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras.utils import np_utils

from keras.models import Sequential

from keras.layers import Dense, Lambda

from keras.layers.core import Dropout

from keras import regularizers

from keras.models import Model, load_model

from keras.preprocessing import sequence

from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation

from keras.layers import Embedding

from keras.layers import Conv1D, GlobalMaxPooling1D

from keras.datasets import imdb

from keras.models import Sequential

from keras.layers import Dense, Activation, MaxPooling1D

cols = [c for c in train.columns if c not in ['is_churn','msno']]



X_train = StandardScaler().fit_transform(train[cols].as_matrix())

y_train = train['is_churn'].as_matrix()

X_test = StandardScaler().fit_transform(test[cols].as_matrix())
lsize = 128

model = Sequential()

model.add(Dense(lsize, input_dim=int(X_train.shape[1]),activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(rate=0.25))

model.add(Dense(int(lsize/2), activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(rate=0.25))

model.add(Dense(int(lsize/4),kernel_regularizer=regularizers.l2(0.1), activation='relu'))

model.add(Dropout(rate=0.1))

model.add(Dense(1, activation='sigmoid'))



# Compile model

model.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])

model.summary()
# Fit the model

history = model.fit(X_train, y_train, epochs=10, batch_size=1026,#512, 

                    validation_split=0.2, verbose=1)

                    
predictions = model.predict(X_test)

test['is_churn'] = predictions.clip(0.+1e-15, 1-1e-15)

test[['msno','is_churn']].to_csv('submission_NN.csv', index=False)