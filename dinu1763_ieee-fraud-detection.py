# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import os



import numpy as np

import pandas as pd

from sklearn import preprocessing

import xgboost as xgb

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#-----------------Get some References-------------------------#



#----------https://www.kaggle.com/artgor/eda-and-models/notebook------------------------#



#----------https://www.kaggle.com/jesucristo/fraud-complete-eda-------------------------#



#----------https://www.kaggle.com/robikscube/ieee-fraud-detection-first-look-and-eda----#



#----------https://www.kaggle.com/artkulak/ieee-fraud-simple-baseline-0-9383-lb---------#



#----------https://www.kaggle.com/mjbahmani/reducing-memory-size-for-ieee---------------#



#----------https://www.kaggle.com/inversion/ieee-simple-xgboost-------------------------#



# https://www.kaggle.com/kimchiwoong/ieee-fraud-detection-prediction
%%time

train_transaction = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_transaction.csv', index_col='TransactionID')

test_transaction = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_transaction.csv', index_col='TransactionID')



train_identity = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_identity.csv', index_col='TransactionID')

test_identity = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_identity.csv', index_col='TransactionID')



#sample_submission = pd.read_csv('../input/sample_submission.csv', index_col='TransactionID')

train = train_transaction.merge(train_identity, how='left', left_index=True, right_index=True)

test = test_transaction.merge(test_identity, how='left', left_index=True, right_index=True)
print(train.shape)

print(test.shape)
y_train = train['isFraud'].copy()

del train_transaction, train_identity, test_transaction, test_identity
# Drop target, fill in NaNs

X_train = train.drop('isFraud', axis=1)

X_test = test.copy()

del train, test
X_train = X_train.fillna(-999)

X_test = X_test.fillna(-999)
# Label Encoding

for f in X_train.columns:

    if X_train[f].dtype=='object' or X_test[f].dtype=='object': 

        lbl = preprocessing.LabelEncoder()

        lbl.fit(list(X_train[f].values) + list(X_test[f].values))

        X_train[f] = lbl.transform(list(X_train[f].values))

        X_test[f] = lbl.transform(list(X_test[f].values))
clf = xgb.XGBClassifier(

    n_estimators=500,

    max_depth=9,

    learning_rate=0.05,

    subsample=0.9,

    colsample_bytree=0.9,

    missing=-999,

    random_state=2019,

    tree_method='gpu_hist'  # THE MAGICAL PARAMETER

)
%time clf.fit(X_train, y_train)
sample_submission = pd.read_csv('/kaggle/input/ieee-fraud-detection/sample_submission.csv', index_col='TransactionID')
sample_submission['isFraud'] = clf.predict_proba(X_test)[:,1]

sample_submission.to_csv('simple_xgboost.csv')