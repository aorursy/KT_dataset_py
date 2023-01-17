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
pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)

pd.set_option('display.width', 800)
train_identity = pd.read_csv('../input/ieee-fraud-detection/train_identity.csv')

#print('The training dataset for identity information has {0} columns and {1} rows'.format(*train_identity.shape))



train_transaction = pd.read_csv('../input/ieee-fraud-detection/train_transaction.csv')

#print('The training dataset for transactions has {0} columns and {1} rows'.format(*train_transaction.shape))
test_identity = pd.read_csv('../input/ieee-fraud-detection/test_identity.csv')

#print('The test dataset for identity information has {0} columns and {1} rows'.format(*test_identity.shape))



test_transaction = pd.read_csv('../input/ieee-fraud-detection/test_transaction.csv')

#print('The test dataset for identity information has {0} columns and {1} rows'.format(*test_transaction.shape))
sample_submission = pd.read_csv('../input/ieee-fraud-detection/sample_submission.csv', index_col='TransactionID')
train_identity.head()
#Merging the identity and transaction columns

train = pd.merge(train_transaction, train_identity, on='TransactionID', how='left')

test = pd.merge(test_transaction, test_identity, on='TransactionID', how='left')
train.head()
#Clearing space by deleting unwanted dataframes

del train_identity, train_transaction, test_identity, test_transaction
train.isnull().any().sum()
test.head()
train.describe()
y_train = train['isFraud'].astype("uint8").copy()
X_train = train.drop('isFraud', axis=1)

X_test = test.copy()
#label encoding

from sklearn.preprocessing import LabelEncoder

for f in X_train.columns:

    if X_train[f].dtype=='object' or X_test[f].dtype=='object': 

        lbl = LabelEncoder()

        lbl.fit(list(X_train[f].values) + list(X_test[f].values))

        X_train[f] = lbl.transform(list(X_train[f].values))

        X_test[f] = lbl.transform(list(X_test[f].values))  
drop_col = ['TransactionDT']

X_train.drop(drop_col,axis=1, inplace=True)

X_test.drop(drop_col, axis=1, inplace=True)

X_train.head()
X_train.set_index('TransactionID', inplace=True)

X_test.set_index('TransactionID', inplace=True)
#Filling NaN values

X_train.fillna(-1,inplace=True)

X_test.fillna(-1,inplace=True)

X_train.head()
#free up the space

del train, test
import xgboost as xgb

xgboost = xgb.XGBClassifier(

        n_estimators=500,

        max_depth=9,

        learning_rate=0.05,

        subsample=0.9,

        colsample_bytree=0.9,

        gamma = 0.3,

        alpha = 4,

        missing = -1,

        random_state=5894,

        max_delta_step=8,

        tree_method='gpu_hist',

)
%time xgboost.fit(X_train, y_train)
from sklearn.metrics import roc_auc_score

roc_auc_score( y_train, xgboost.predict_proba(X_train)[:,1] )
sample_submission['isFraud'] = xgboost.predict_proba(X_test)[:,1]

sample_submission.to_csv('_xgboost.csv')