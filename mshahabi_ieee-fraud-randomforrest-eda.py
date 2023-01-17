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


import matplotlib.pyplot as plt

import datetime

from sklearn.preprocessing import LabelEncoder

from tqdm import tqdm
train_transaction = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_transaction.csv')

train_transaction.info()

test_transaction = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_transaction.csv')

test_transaction.info()





train_identity = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_identity.csv')

train_identity.info()

test_identity = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_identity.csv')

test_identity.info()



startdate = datetime.datetime.strptime('2017-12-01', '%Y-%m-%d')

train_transaction['TransactionDT'] = train_transaction['TransactionDT'].apply(lambda x: (startdate + datetime.timedelta(seconds = x)))



sample_submission = pd.read_csv('/kaggle/input/ieee-fraud-detection/sample_submission.csv')


# Check the number of transaction per day

total_trans_per_day = train_transaction.groupby(train_transaction['TransactionDT'].dt.date)['isFraud'].count()

train_transaction.groupby(train_transaction['TransactionDT'].dt.date)['isFraud'].count().plot()



#plot the number of fraduelnt transactions per day

total_fraud_per_day = train_transaction[train_transaction['isFraud']==1].groupby(train_transaction['TransactionDT'].dt.date)['isFraud'].count()

train_transaction[train_transaction['isFraud']==1].groupby(train_transaction['TransactionDT'].dt.date)['isFraud'].count().plot()
fraud_ratio_per_day = pd.merge(total_trans_per_day,

                 total_fraud_per_day,

                 on='TransactionDT', 

                 how='left')

fraud_ratio_per_day['fraud_ratio']=100*fraud_ratio_per_day['isFraud_y']/fraud_ratio_per_day['isFraud_x']

fraud_ratio_per_day['fraud_ratio'].plot()
#average transaction amount per day if the transaction is fragulant

train_transaction[train_transaction['isFraud']==1].groupby(train_transaction['TransactionDT'].dt.date)['TransactionAmt'].mean().plot()

#average transaction amount per day if the transaction is non-fragulant

train_transaction[train_transaction['isFraud']==0].groupby(train_transaction['TransactionDT'].dt.date)['TransactionAmt'].mean().plot()



train_transaction[train_transaction['isFraud']==0].groupby(train_transaction['ProductCD'])['isFraud'].count().plot(kind='bar')
########## CALCULATE THE FEATURE IMOPRTANCE USING DECISION TREES

# First we merge transaction and identity datasets on key = 

train_transaction_all = pd.merge(train_transaction,

                 train_identity,

                 on='TransactionID', 

                 how='left')



test_transaction_all = pd.merge(test_transaction,

                 test_identity,

                 on='TransactionID', 

                 how='left')



cat_data = train_transaction_all.select_dtypes(include='object')

cat_cols = list(cat_data.columns.values)



#train_transaction_all_orig = train_transaction_all.copy()

#test_transaction_all_orig  = test_transaction_all.copy()



train_data = train_transaction_all.copy()

train_data = train_data.drop(columns=['isFraud'])
del train_transaction

del test_transaction

del test_identity

del train_identity
for i in tqdm(cat_cols): 

    label = LabelEncoder()

    label.fit(list(train_data[i].values)+list(test_transaction_all[i].values))

    train_data[i] = label.transform(list(train_data[i].values))

    test_transaction_all[i]  = label.transform(list(test_transaction_all[i].values))









X = train_data.drop(columns=['TransactionID','TransactionDT']).fillna(-999)

X_test = test_transaction_all.drop(columns=['TransactionID','TransactionDT']).fillna(-999)

y = train_transaction_all['isFraud']
del train_transaction_all
from sklearn.datasets import make_classification

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import log_loss,accuracy_score, f1_score,roc_auc_score, confusion_matrix

def get_performance(y, predictions_proba,predictions):

    log_loss_score = log_loss(y, predictions_proba)

    acc = accuracy_score(y, predictions)

    f1 = f1_score(y, predictions)

    print('AUC',roc_auc_score(y,predictions))

    print('Log loss: %.5f' % log_loss_score)  # 0.62923

    print('Acc: %.5f' % acc)  # 0.70952

    print('F1: %.5f' % f1)  # 0.59173

    print(confusion_matrix(y, predictions))

# Build a classification task 

clf = RandomForestClassifier(random_state=0, n_jobs=-1)

model = clf.fit(X, y)

predictions_proba = model.predict_proba(X)

predictions = model.predict(X)

get_performance(y, predictions_proba,predictions)
sample_submission['isFraud']=y
sample_submission.to_csv('out.csv', index=False)