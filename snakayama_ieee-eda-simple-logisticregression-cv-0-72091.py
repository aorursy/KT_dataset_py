# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import os

import warnings

import gc

import time

from tqdm import tqdm

import functools

from sklearn import preprocessing

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import StratifiedKFold

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_auc_score

from sklearn.preprocessing import LabelEncoder
def graph_insight(data):

    print(set(data.dtypes.tolist()))

    df_num = data.select_dtypes(include = ['float64', 'int64'])

    df_num.hist(figsize=(16, 16), bins=50, xlabelsize=8, ylabelsize=8);



def read_csv(path):

  # logger.debug('enter')

  df = pd.read_csv(path)

  # logger.debug('exit')

  return df



def load_train_data():

  # logger.debug('enter')

  df = read_csv(SALES_TRAIN_V2)

  # logger.debug('exit')

  return df



def load_test_data():

  # logger.debug('enter')

  df = read_csv(TEST_DATA)

  # logger.debug('exit')

  return df



def graph_insight(data):

    print(set(data.dtypes.tolist()))

    df_num = data.select_dtypes(include = ['float64', 'int64'])

    df_num.hist(figsize=(16, 16), bins=50, xlabelsize=8, ylabelsize=8);



def drop_duplicate(data, subset):

    print('Before drop shape:', data.shape)

    before = data.shape[0]

    data.drop_duplicates(subset,keep='first', inplace=True) #subset is list where you have to put all column for duplicate check

    data.reset_index(drop=True, inplace=True)

    print('After drop shape:', data.shape)

    after = data.shape[0]

    print('Total Duplicate:', before-after)



def unresanable_data(data):

    print("Min Value:",data.min())

    print("Max Value:",data.max())

    print("Average Value:",data.mean())

    print("Center Point of Data:",data.median())
SAMPLE_SUBMISSION    = '../input/sample_submission.csv'

TRAIN_DATA           = '../input/train_identity.csv'

TRAIN_TR_DATA        = '../input/train_transaction.csv'

TEST_DATA            = '../input/test_identity.csv'

TEST_TR_DATA         = '../input/test_transaction.csv'



df_sample            = read_csv(SAMPLE_SUBMISSION)

df_train_id          = read_csv(TRAIN_DATA)

df_train_tr          = read_csv(TRAIN_TR_DATA)

df_test_id           = read_csv(TEST_DATA)

df_test_tr           = read_csv(TEST_TR_DATA)
graph_insight(df_train_id)
df_train_tr['isFraud'].hist(bins =4)
df_train_tr.query('isFraud==0').shape
df_train_tr.query('isFraud==1').shape
df_train = pd.merge(df_train_tr, df_train_id, how='left', on='TransactionID')

df_test = pd.merge(df_test_tr, df_test_id, how='left', on='TransactionID')
df_train.shape
df_test.shape
df_sample.shape
df_train = df_train.fillna(0)

df_test = df_test.fillna(0)



cat_cols = ['id_12', 'id_13', 'id_14', 'id_15', 'id_16', 'id_17', 'id_18', 'id_19', 'id_20', 'id_21', 'id_22', 'id_23', 'id_24', 'id_25', 'id_26', 'id_27', 'id_28', 'id_29',

            'id_30', 'id_31', 'id_32', 'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38', 'DeviceType', 'DeviceInfo', 'ProductCD', 'card4', 'card6', 'M4','P_emaildomain',

            'R_emaildomain', 'card1', 'card2', 'card3',  'card5', 'addr1', 'addr2', 'M1', 'M2', 'M3', 'M5', 'M6', 'M7', 'M8', 'M9']

for col in cat_cols:

    if col in df_train.columns:

        le = LabelEncoder()

        le.fit(list(df_train[col].astype(str).values) + list(df_test[col].astype(str).values))

        df_train[col] = le.transform(list(df_train[col].astype(str).values))

        df_test[col] = le.transform(list(df_test[col].astype(str).values))   
df_train.head()
def reduce_mem_usage(df):

    start_mem_usg = df.memory_usage().sum() / 1024**2 

    print("Memory usage of properties dataframe is :",start_mem_usg," MB")

    NAlist = [] # Keeps track of columns that have missing values filled in. 

    for col in df.columns:

        if df[col].dtype != object:  # Exclude strings            

            # Print current column type

            print("******************************")

            print("Column: ",col)

            print("dtype before: ",df[col].dtype)            

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

            

            # Print new column type

            print("dtype after: ",df[col].dtype)

            print("******************************")

    # Print final result

    print("___MEMORY USAGE AFTER COMPLETION:___")

    mem_usg = df.memory_usage().sum() / 1024**2 

    print("Memory usage is: ",mem_usg," MB")

    print("This is ",100*mem_usg/start_mem_usg,"% of the initial size")

    return df, NAlist
df_train, NAlist = reduce_mem_usage(df_train)

df_test, NAlist_t = reduce_mem_usage(df_test)
cols = [c for c in df_train.columns if c not in ['TransactionID', 'isFraud']]

oof = np.zeros(len(df_train))

preds = np.zeros(len(df_test))
skf = StratifiedKFold(n_splits=25, random_state=42)

for train_index, test_index in skf.split(df_train.iloc[:,1:-1], df_train['isFraud']):

    clf = LogisticRegression(solver='liblinear',penalty='l2',C=1.0)

    clf.fit(df_train.loc[train_index][cols],df_train.loc[train_index]['isFraud'])

    oof[test_index] = clf.predict_proba(df_train.loc[test_index][cols])[:,1]

    preds += clf.predict_proba(df_test[cols])[:,1] / 25.0

    

auc = roc_auc_score(df_train['isFraud'],oof)

print('LR without interactions scores CV =',round(auc,5))
df_sample['isFraud'] = preds

df_sample.to_csv('submission.csv',index=False)