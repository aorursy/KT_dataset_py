import pandas as pd

import numpy as np



# Load data

train = pd.read_csv('../input/cat-in-the-dat/train.csv')

test = pd.read_csv('../input/cat-in-the-dat/test.csv')



target = train['target']

train_id = train['id']

test_id = test['id']

train.drop(['target', 'id'], axis=1, inplace=True)

test.drop('id', axis=1, inplace=True)



print(train.shape)

print(test.shape)
%%time



# STEP 1

traintest = pd.concat([train, test])

dummies = pd.get_dummies(traintest, columns=traintest.columns, drop_first=True, sparse=True)

train_ohe = dummies.iloc[:train.shape[0], :]

test_ohe = dummies.iloc[train.shape[0]:, :]



print(train_ohe.shape)

print(test_ohe.shape)
%%time



# STEP 2

train_ohe = train_ohe.sparse.to_coo().tocsr()

test_ohe = test_ohe.sparse.to_coo().tocsr()
%%time



# STEP 1.1

traintest = pd.concat([train, test])

dummies = pd.get_dummies(traintest, columns=traintest.columns, drop_first=True, sparse=True)



print(dummies.shape)
%%time



# STEP 1.2

train_ohe = dummies.iloc[:train.shape[0], :]

test_ohe = dummies.iloc[train.shape[0]:, :]



print(train_ohe.shape)

print(test_ohe.shape)
%%time



traintest = pd.concat([train, test])

dummies = pd.get_dummies(traintest, columns=traintest.columns, drop_first=True, sparse=True)



dummies_csr = dummies.sparse.to_coo().tocsr()



train_ohe = dummies_csr[:train.shape[0], :]

test_ohe = dummies_csr[train.shape[0]:, :]



print(train_ohe.shape)

print(test_ohe.shape)