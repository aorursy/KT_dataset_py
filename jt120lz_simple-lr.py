# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import multiprocessing

import warnings

import matplotlib.pyplot as plt

import seaborn as sns

import lightgbm as lgb

import gc

from time import time

import datetime

from tqdm import tqdm_notebook

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import KFold, TimeSeriesSplit

from sklearn.metrics import roc_auc_score

warnings.simplefilter('ignore')



from sklearn.preprocessing import OneHotEncoder, StandardScaler

from scipy import sparse as ss

from sklearn.linear_model import LogisticRegression
files = ['../input/ieee-fraud-detection/test_identity.csv', 

         '../input/ieee-fraud-detection/test_transaction.csv',

         '../input/ieee-fraud-detection/train_identity.csv',

         '../input/ieee-fraud-detection/train_transaction.csv',

         '../input/ieee-fraud-detection/sample_submission.csv']
%%time

def load_data(file):

    return pd.read_csv(file)



with multiprocessing.Pool() as pool:

    test_id, test_tr, train_id, train_tr, sub = pool.map(load_data, files)
train = pd.merge(train_tr, train_id, on='TransactionID', how='left')

test = pd.merge(test_tr, test_id, on='TransactionID', how='left')



del test_id, test_tr, train_id, train_tr

gc.collect()
startdate = datetime.datetime.strptime('2017-11-30', '%Y-%m-%d')

train['TransactionDT'] = train['TransactionDT'].apply(lambda x: (startdate + datetime.timedelta(seconds = x)))
train = train[train['TransactionDT']>'2018-04-01']

print(train.shape)
not_feature_cols = ['TransactionID','isFraud','TransactionDT']

obj_cols = []

for c in train.columns:

    if train[c].dtype=='object':

        obj_cols.append(c)

card_cols = ['card' + str(i) for i in range(1,6)]

id_cols = ['id_14']

obj_cols = obj_cols + card_cols + id_cols

print('object columns:', len(obj_cols), ','.join(obj_cols))



num_cols = [x for x in train.columns if x not in not_feature_cols + obj_cols]

print('number columns:', len(num_cols), ','.join(num_cols))







t1 = train[num_cols]

t1 = t1.astype('float32')

t1.fillna(0, inplace=True)

st = StandardScaler()

t1 = st.fit_transform(t1)



t2 = train[obj_cols]

# t2.fillna('na', inplace=True)

t2 = t2.astype('str')

e1 = OneHotEncoder(handle_unknown='ignore')

e1.fit(t2)

t2 = e1.transform(t2)

x = ss.hstack((t1, t2))

print(x.shape)

lr = LogisticRegression()

lr.fit(x, train['isFraud'])
print(len(test))

t1 = test[num_cols]

t1 = t1.astype('float32')

t1.fillna(0, inplace=True)

t1 = st.transform(t1)



t2 = test[obj_cols]

t2 = t2.astype('str')

t2 = e1.transform(t2)

test_x = ss.hstack((t1, t2))

print(test_x.shape)
pred = lr.predict_proba(test_x)[:, 1]

result = test[['TransactionID']]

result['isFraud'] = pred

result.to_csv('submission.csv', index=None)