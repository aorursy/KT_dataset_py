import numpy as np

import pandas as pd

import os

import gc



path = '/kaggle/input/ieee-fraud-detection/'



# train

train_identity = pd.read_csv(f'{path}train_identity.csv')

train_transaction = pd.read_csv(f'{path}train_transaction.csv')

train = pd.merge(train_transaction, train_identity, on='TransactionID', how='left', left_index=True, right_index=True)

del train_identity, train_transaction

gc.collect()



# test

test_identity = pd.read_csv(f'{path}test_identity.csv')

test_transaction = pd.read_csv(f'{path}test_transaction.csv')

test = pd.merge(test_transaction, test_identity, on='TransactionID', how='left', left_index=True, right_index=True)

del test_identity, test_transaction

gc.collect()



# tmp

tmp = pd.concat([train, test], sort=False, axis=0, keys=['train', 'test'])

del train, test

gc.collect()



tmp['isFraud'] = tmp['isFraud'].fillna(0)
tmp.sample(20).sort_index()
tmp.to_pickle('concat.pkl')