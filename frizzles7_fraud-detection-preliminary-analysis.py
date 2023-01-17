# imports and list of available datafiles

import numpy as np

import pandas as pd

from sklearn import preprocessing

import xgboost as xgb

from sklearn.model_selection import KFold

from sklearn.metrics import roc_auc_score

import gc

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# read the files into dataframes

train_iden = pd.read_csv('../input/train_identity.csv', index_col='TransactionID')

train_tran = pd.read_csv('../input/train_transaction.csv', index_col='TransactionID')

test_iden = pd.read_csv('../input/test_identity.csv', index_col='TransactionID')

test_tran = pd.read_csv('../input/test_transaction.csv', index_col='TransactionID')
train_iden.head()
train_tran.head()
# merge transactions and identities

train = train_tran.merge(train_iden, how='left', left_index=True, right_index=True, sort=False)

test = test_tran.merge(test_iden, how='left', left_index=True, right_index=True, sort=False)
# drop dataframes no longer needed

del train_tran, train_iden, test_tran, test_iden

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
# see https://www.kaggle.com/mjbahmani/reducing-memory-size-for-ieee

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
# baseline

kf = KFold(n_splits=3, shuffle=True, random_state=0)

scores = 0

for train_idx, valid_idx in kf.split(X_train, y_train):

    clf = xgb.XGBClassifier(n_estimators=500,

                            n_jobs=4,

                            max_depth=9,

                            learning_rate=0.05,

                            subsample=0.8,

                            colsample_bytree=0.8,

                            missing=-999, 

                            gamma=0.1,

                            tree_method='gpu_hist',

                            )

    

    X_tra, X_val = X_train.iloc[train_idx, :], X_train.iloc[valid_idx, :]

    y_tra, y_val = y_train.iloc[train_idx], y_train.iloc[valid_idx]

    clf.fit(X_tra, y_tra)

    preds = clf.predict_proba(X_val)[:,1]

    auc_score = roc_auc_score(y_val, preds)

    scores += auc_score

scores /= 3

print('Avg ROC AUC score: ', scores)
gc.collect()
# train final XGB classifier

clf = xgb.XGBClassifier(n_estimators=500,

                        n_jobs=4,

                        max_depth=9,

                        learning_rate=0.05,

                        subsample=0.8,

                        colsample_bytree=0.8,

                        missing=-999, 

                        gamma=0.1,

                        tree_method='gpu_hist',

                        )



clf.fit(X_train, y_train)
# predicted probabilities to file

sample_submission = pd.read_csv('../input/sample_submission.csv', index_col='TransactionID')

sample_submission['isFraud'] = clf.predict_proba(X_test)[:,1]

sample_submission.to_csv('sample_xgb.csv')