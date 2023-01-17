# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
## IEEE - project
identity_train = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_identity.csv')
transaction_train = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_transaction.csv')
identity_test = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_identity.csv')
transaction_test = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_transaction.csv')
print('identity_train: ', identity_train.shape)
print('transaction_train: ', transaction_train.shape)
print('identity_test: ', identity_test.shape)
print('transaction_test: ', transaction_test.shape)
identity_train.columns
identity_train.head()
transaction_train.head()
transaction_train['isFraud'].value_counts().to_frame()
train = pd.merge(identity_train, transaction_train, on='TransactionID')
train.shape
train.isnull().sum()
total = np.product(train.shape)
empty = train.isnull().sum().sum()
per  = empty/total
print('Total records:', total,'\n' 'Total null records:',  empty, '\n' 'Percentage of null values:',empty/total)
train.isnull().mean()
full_columns = [c for c in train.columns if train[c].isnull().mean() == 0 ]
full_columns
len(full_columns)
train_full = train[full_columns].copy()
train_full.sample(10)
train_full.shape
train_full.isnull().mean()
sns.distplot(a=train_full['TransactionAmt'])
# sns.heatmap(data=train_full, annot=True)
continuous_variables = [i for i in train_full.columns if train_full[i].dtype in ['int', 'float']]
continuous_variables
len(continuous_variables)
train[continuous_variables].isnull().mean()
categorical_variables = [i for i in train_full.columns if train_full[i].dtype == 'object']
categorical_variables
train_full[categorical_variables].isnull().mean()
train_full.id_12.unique()
train_full.ProductCD.unique()
