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
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.model_selection import train_test_split

from sklearn.impute import SimpleImputer
train_identity = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_identity.csv',index_col='TransactionID')

test_identity  = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_identity.csv',index_col='TransactionID')

train_transaction = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_transaction.csv',index_col='TransactionID')

test_transaction  = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_transaction.csv',index_col='TransactionID')

train_identity.head()
print("Shape of Train_identity:: ", train_identity.shape)

print("Shape of Test_identity:: ", test_identity.shape)

print("Shape of Train_transaction:: ", train_transaction.shape)

print("Shape of Test_transaction:: ", test_transaction.shape)
train_transaction.head()
train_transaction.info()
train_transaction.describe()
train_transaction.reset_index()['TransactionID'].isin(train_identity.reset_index()['TransactionID']).value_counts()
train_identity.reset_index()['TransactionID'].isin(train_transaction.reset_index()['TransactionID']).value_counts()
X = pd.merge(train_transaction,

             train_identity,

             on='TransactionID',

             how='left')

print("train_transaction dimensions: {} ".format(train_transaction.shape))

print("train_identity dimensions:    {} ".format(train_identity.shape))

print("Merged X dimensions:          {} ".format(X.shape))
X.head()
Y = X.isFraud

X = X.reset_index().drop('isFraud', axis=1)
X.head()
#Splitting bthe data to train and validation set

X_train, X_valid, y_train, y_valid = train_test_split(X, Y, test_size=0.10, random_state=42)
print("X_train dimensions:  {}".format(X_train.shape))

print("y_train dimensions:  {}".format(y_train.shape))

print("X_valid dimensions:  {}".format(X_valid.shape))

print("y_valid dimensions:  {}".format(y_valid.shape))
numeric_columns = list(X_train.select_dtypes(exclude='object').columns)

categorical_columns = list(X_train.select_dtypes(include='object').columns)
categorical_columns
columns = ['card1', 'card2', 'card3', 'card5', 'addr1', 'addr2', 'ProductCD',

 'card4',

 'card6',

 'P_emaildomain',

 'R_emaildomain',

 'M1',

 'M2',

 'M3',

 'M4',

 'M5',

 'M6',

 'M7',

 'M8',

 'M9',

 'id_12',

 'id_15',

 'id_16',

 'id_23',

 'id_27',

 'id_28',

 'id_29',

 'id_30',

 'id_31',

 'id_33',

 'id_34',

 'id_35',

 'id_36',

 'id_37',

 'id_38',

 'DeviceType',

 'DeviceInfo']



for col in columns:

    X_train[col] = X_train[col].astype('category')

    X_valid[col] = X_valid[col].astype('category')
numeric_columns = list(X_train.select_dtypes(exclude='category').columns)

categorical_columns = list(X_train.select_dtypes(include='category').columns)
categorical_columns
print("Numeric Missing Numbers: {}".format(X_train[numeric_columns].isnull().sum()[X_train[numeric_columns].isnull().sum() > 0]))
print("Categorical Missing Numbers: {}".format(X_train[categorical_columns].isnull().sum()[X_train[categorical_columns].isnull().sum() > 0]))
X_train[categorical_columns].nunique()
good_category_columns  = [col for col in categorical_columns if set(X_train[col]) == set(X_valid[col])]

bad_category_columns  = list(set(categorical_columns) - set(good_category_columns))

bad_category_columns