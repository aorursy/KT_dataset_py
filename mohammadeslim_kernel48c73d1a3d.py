import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

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

sns.set()

%matplotlib inline

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_identity=pd.read_csv("/kaggle/input/ieee-fraud-detection/train_identity.csv")

train_transaction=pd.read_csv("/kaggle/input/ieee-fraud-detection/train_transaction.csv")

test_identity=pd.read_csv("/kaggle/input/ieee-fraud-detection/test_identity.csv")

test_transaction=pd.read_csv("/kaggle/input/ieee-fraud-detection/test_transaction.csv")
def reduce_mem_usage(df, verbose=True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024**2    

    for col in df.columns:

        col_type = df[col].dtypes

        if col_type in numerics:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)    

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df
train=pd.merge(train_transaction,train_identity,how="left",on="TransactionID")

test=pd.merge(test_transaction,test_identity,how="left",on="TransactionID")
train=reduce_mem_usage(train)

test=reduce_mem_usage(test)
del train_identity

del test_identity

del train_transaction

del test_transaction
train.head(5)
more_than_90_NA_or_same_value_train=[]

more_than_90_NA_or_same_value_test=[]

many_na_train=[]

many_na_test=[]

for col in train.columns:

    if train[col].isna().sum()/train.shape[0] >=0.90:

        many_na_train.append(col) # full of NAs in train

for col in test.columns:

    if test[col].isna().sum()/test.shape[0]>=0.90:

        many_na_test.append(col) # full of NAs in test

for col in train.columns:

  #  print(col,train[col].value_counts(dropna=False,normalize=True).values[0])

    if train[col].value_counts(dropna=False,normalize=True).values[0] >= 0.90:

      #  print("More than 90% is NA's or same value so we can delete that columns")

        more_than_90_NA_or_same_value_train.append(col) # more unique values in train

for col in test.columns:

    if test[col].value_counts(dropna=False,normalize=True).values[0]>=0.90:

        more_than_90_NA_or_same_value_test.append(col) #more unique values in test
# store the columns to be dropped separately in train and test

cols_drop_at_train=list(set(more_than_90_NA_or_same_value_train+many_na_train))

cols_drop_at_test=list(set(more_than_90_NA_or_same_value_test+many_na_test))

print("Columns to be dropped in train",len(cols_drop_at_train))

print("Columns to be dropped in test",len(cols_drop_at_test))

print("columns are @ train:",cols_drop_at_train)

print("columns are @ test:", cols_drop_at_train)
total_drop_cols=list(set(cols_drop_at_train+cols_drop_at_test))

print("Total no of columns to be deleted to increase your model performance",len(total_drop_cols))

print("They are:",total_drop_cols)
# remove the isFraud

total_drop_cols.remove('isFraud')

print("You can check thta column is removed:",total_drop_cols)
train.drop(total_drop_cols, axis=1)

test.drop(total_drop_cols, axis=1)

print(len(train.columns))
sns.distplot(train['TransactionDT'], hist=True, kde=True,bins=40) # its shows histogram along with the density plot

sns.distplot(test['TransactionDT'],hist=True,kde=True,bins=40)

plt.title('Density Plot of  TransactionDT  in training data')

plt.xlabel(' TransactionDT')

plt.ylabel('Counts')