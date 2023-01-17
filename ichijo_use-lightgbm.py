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
from sklearn.preprocessing import LabelEncoder

import lightgbm as lgb

import time
pd.set_option('display.max_columns', 100)

pd.set_option('display.width', 200)

ts = time.time()
train_transaction = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_transaction.csv')

train_identity = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_identity.csv')

test_transaction = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_transaction.csv')

test_identity = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_identity.csv')

sample_submission = pd.read_csv('/kaggle/input/ieee-fraud-detection/sample_submission.csv')

sample_submission
del sample_submission
train_transaction.head()
train_identity.head()
test_transaction.head()
test_identity.head()
train = pd.merge(train_transaction, train_identity, on='TransactionID', how='left')

test = pd.merge(test_transaction, test_identity, on='TransactionID', how='left')
del train_transaction, train_identity, test_transaction, test_identity
train.isna().all().sum(), train.isna().all(axis=1).sum()
# train, test columns are slightly different

cols = list(train.columns)

cols.remove('isFraud')

test.columns = cols
# Remove columns that have few features in the data

big_top_volume_cols = [col for col in train.columns if train[col].value_counts(dropna=False, normalize=True).values[0] > 0.9]

big_top_volume_cols.remove('isFraud')
train = train.drop(big_top_volume_cols, axis=1)

test = test.drop(big_top_volume_cols, axis=1)
# Encode categorical features

cat_cols = ['ProductCD', 'card4', 'card6', 'P_emaildomain', 'R_emaildomain', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'id_12', 'id_15', 'id_16', 'id_28', 'id_29', 'id_30',

            'id_31', 'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38', 'DeviceType', 'DeviceInfo']

for col in cat_cols:

    le = LabelEncoder()

    col_val_train = list(train[col].astype(str).values)

    col_val_test = list(test[col].astype(str).values)

    col_val = col_val_train + col_val_test

    le.fit(col_val)

    train[col] = le.transform(col_val_train)

    test[col] = le.transform(col_val_test)
train = train.sort_values('TransactionDT')

y_train = train['isFraud']

X_train = train.drop(['TransactionDT', 'TransactionID', 'isFraud'], axis=1)

test = test.sort_values('TransactionDT')

X_test = test.drop(['TransactionDT', 'TransactionID'], axis=1)
X_train = X_train.fillna(0)

X_test = X_test.fillna(0)

sub = test['TransactionID']
train_data = lgb.Dataset(X_train, label=y_train)

params = {

    'task': 'train',

    'boosting_type': 'gbdt',

    'objective': 'binary',

}

gbm = lgb.train(

    params,

    train_data,

    num_boost_round=100,

)

y_pred = gbm.predict(X_test)
sub = pd.concat([sub, pd.Series(y_pred, name='isFraud')], axis=1)

print('isFraud probability is: ' + str(sub['isFraud'].mean()))

sub.to_csv('submission.csv', index=False)