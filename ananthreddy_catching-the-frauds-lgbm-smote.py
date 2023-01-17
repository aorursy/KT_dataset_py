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
import sys,re



import matplotlib.pyplot as plt

import seaborn as sns

import pandas_profiling

import warnings



from tqdm import tqdm

warnings.filterwarnings('ignore')
train_identity = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_identity.csv')

print (train_identity.shape)

train_identity.head()
train_transaction = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_transaction.csv')

print (train_transaction.shape)

train_transaction.head()
test_identity = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_identity.csv')

print (test_identity.shape)

test_identity.head()
test_transaction = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_transaction.csv')

print (test_transaction.shape)

test_transaction.head()
def train_data_processing(transaction_data, identity_data):

    data = pd.merge(transaction_data, identity_data, on='TransactionID', how = 'left', suffixes=('_x','_y'))

    del_columns = []

    for column in data.columns:

        if ((data[column].isnull().sum()) > (len(data)*0.60)):

            del_columns.append(column)

    catcols = ['ProductCD','card4','M4','V14','V41','V65','V88','V94','card6','M1', 'M2', 'M3', 'M5', 'M6', 'M7', 'M8', 'M9']

    del_columns_1 = ['addr1','addr2','C5','C9','V12','V13','V29','V30','V35','V36','V48','V49','V53','V54','V69','V70','V75','V76','V90','V91','V107','V305','P_emaildomain']

    

    del_columns.extend(del_columns_1)

    data = data.drop(columns = del_columns, axis = 1)

    # data['isFraud'] = data['isFraud'].apply(lambda x: -1 if x ==0 else 1)

    data = pd.get_dummies(columns = catcols, data = data)

    return data
train_data = train_data_processing(train_transaction, train_identity)
def test_data_processing(transaction_data, identity_data):

    data = pd.merge(transaction_data, identity_data, on='TransactionID', how = 'left', suffixes=('_x','_y'))

    del_columns = []

    for column in data.columns:

        if ((data[column].isnull().sum()) > (len(data)*0.60)):

            del_columns.append(column)

    catcols = ['ProductCD','card4','M4','V14','V41','V65','V88','V94','card6','M1', 'M2', 'M3', 'M6', 'M7', 'M8', 'M9']

    del_columns_1 = ['addr1','addr2','C5','C9','V12','V13','V29','V30','V35','V36','V48','V49','V53','V54','V69','V70','V75','V76','V90','V91','V107','V305','P_emaildomain']

    

    del_columns.extend(del_columns_1)

    data = data.drop(columns = del_columns, axis = 1)

    data = pd.get_dummies(columns = catcols, data = data)

    return data



test_data = test_data_processing(test_transaction, test_identity)
train_data.head()
test_data.head()
y_train = train_data['isFraud']

X_train = train_data.drop(['isFraud','TransactionID'], 1)



X_test = test_data.drop(['TransactionID'], 1)
X_test['card6_debit or credit'] = 0

X_test['M5_F'] = 0

X_test['M5_T'] = 0
from imblearn.over_sampling import SMOTE



smote = SMOTE(n_jobs = -1)

smote
X_train.mode()
X_train_mode= X_train.mode()

for col in X_train.columns.values:

    X_train[col] = X_train[col].fillna(value = X_train_mode[col].iloc[0])
X_train.isnull().sum()
X_train, y_train = smote.fit_resample(X_train, y_train)
from sklearn.metrics import confusion_matrix, classification_report

from lightgbm import LGBMClassifier

import lightgbm as lgb



lgbmclassifier = LGBMClassifier().fit(X_train, y_train)

lgbm_smote_result = lgbmclassifier.predict_proba(X_test)
lgbm_smote_result[:,1]
submission = pd.read_csv('/kaggle/input/ieee-fraud-detection/sample_submission.csv')

print (submission.shape)

submission.head()
submission['isFraud'] = lgbm_smote_result[:,1]
submission.head()
submission.to_csv('submission_smote_lgbm.csv', index = False)
