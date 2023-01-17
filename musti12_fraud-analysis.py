# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

from sklearn import preprocessing

import xgboost as xgb

from sklearn.model_selection import KFold

from sklearn.metrics import roc_auc_score

import gc



# Any results you write to the current directory are saved as output.
# read the files into dataframes

train_id = pd.read_csv('../input/train_identity.csv', index_col='TransactionID')

train_tr = pd.read_csv('../input/train_transaction.csv', index_col='TransactionID')

test_id = pd.read_csv('../input/test_identity.csv', index_col='TransactionID')

test_tr = pd.read_csv('../input/test_transaction.csv', index_col='TransactionID')
# show first 5 rows of train identity

train_id.head()



# show first 5 rows of train transaction

train_tr.head()

# merge transactions and identities

train = train_tr.merge(train_id, how='left', left_index=True, right_index=True, sort=False)

test = test_tr.merge(test_id, how='left', left_index=True, right_index=True, sort=False)
del train_tr, train_id, test_tr, test_id

gc.collect()
# what portion of our data is fraud?

train['isFraud'].mean()
# categorical columns

cat_cols = ['ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6', 'addr1', 'addr2',

            'P_emaildomain', 'R_emaildomain', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 

            'DeviceType', 'DeviceInfo', 'id_12', 'id_13', 'id_14', 'id_15', 'id_16', 'id_17', 

            'id_18', 'id_19', 'id_20', 'id_21', 'id_22', 'id_23', 'id_24', 'id_25', 'id_26', 'id_27',

            'id_28', 'id_29', 'id_30', 'id_31', 'id_32', 'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 

            'id_38']
# fill missing values

train = train.fillna(-999)

test = test.fillna(-999)
# use label encoding

for col in cat_cols:

    le = preprocessing.LabelEncoder()

    le.fit(list(train[col].values) + list(test[col].values))

    train[col] = le.transform(list(train[col].values))

    test[col] = le.transform(list(test[col].values))


def reduce_mem_usage(df):

    start_mem_usg = df.memory_usage().sum() / 1024**2 

    print("Memory usage of properties dataframe is :",start_mem_usg," MB")

    NAlist = [] # Keeps track of columns that have missing values filled in. 

    for col in df.columns:

        if df[col].dtype != object:  # Exclude strings                        

            # make variables for Int, max and min

            IsInt = False

            mx = df[col].max()

            mn = df[col].min()            

            # Integer does not support NA, therefore, NA needs to be filled

            if not np.isfinite(df[col]).all(): 

                NAlist.append(col)

                df[col].fillna(mn-1,inplace=True)         

            # test if column can be converted to an integer

            asint = df[col].fillna(0).astype(np.int64)

            result = (df[col] - asint)

            result = result.sum()

            if result > -0.01 and result < 0.01:

                IsInt = True            

            # Make Integer/unsigned Integer datatypes

            if IsInt:

                if mn >= 0:

                    if mx < 255:

                        df[col] = df[col].astype(np.uint8)

                    elif mx < 65535:

                        df[col] = df[col].astype(np.uint16)

                    elif mx < 4294967295:

                        df[col] = df[col].astype(np.uint32)

                    else:

                        df[col] = df[col].astype(np.uint64)

                else:

                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:

                        df[col] = df[col].astype(np.int8)

                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:

                        df[col] = df[col].astype(np.int16)

                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:

                        df[col] = df[col].astype(np.int32)

                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:

                        df[col] = df[col].astype(np.int64)    

            # Make float datatypes 32 bit

            else:

                df[col] = df[col].astype(np.float32)

    # Print final result

    print("___MEMORY USAGE AFTER COMPLETION:___")

    mem_usg = df.memory_usage().sum() / 1024**2 

    print("Memory usage is: ",mem_usg," MB")

    print("This is ",100*mem_usg/start_mem_usg,"% of the initial size")

    return df, NAlist

train, NAtrain = reduce_mem_usage(train)

test, NAtest = reduce_mem_usage(test)

# prepare X_train, X_test, y_train

y_train = train['isFraud'].copy()

X_train = train.drop(['isFraud'], axis=1)

X_test = test.copy()



del train, test, cat_cols, le, NAtrain, NAtest

gc.collect()