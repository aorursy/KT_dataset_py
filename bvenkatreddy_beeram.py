# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import gc

import warnings

import sys

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
def reduce_mem_usage_func(df):

    start_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))



    for col in df.columns:

        col_type = df[col].dtype



        if col_type != object:

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

    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))

    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df

   
train_iden = pd.read_csv("../input/ieee-fraud-detection/train_identity.csv", index_col = 'TransactionID')

train_trans = pd.read_csv("../input/ieee-fraud-detection/train_transaction.csv", index_col = 'TransactionID')

test_iden = pd.read_csv("../input/ieee-fraud-detection/test_identity.csv" , index_col = 'TransactionID')

test_trans = pd.read_csv("../input/ieee-fraud-detection/sample_submission.csv" , index_col = 'TransactionID')
train_ide = reduce_mem_usage_func(train_iden)

train_tran = reduce_mem_usage_func(train_trans)

test_tran = reduce_mem_usage_func(test_trans)

test_ide = reduce_mem_usage_func(test_iden)

del train_trans , test_trans

del train_iden , test_iden

gc.collect()
train = train_tran.merge(train_ide, how='left', left_index=True, right_index=True)

test = test_tran.merge(test_ide , how = 'left', left_index = True, right_index = True)

del train_tran, train_ide, test_ide, test_tran

gc.collect()

train.head(10)
train["id_31"].fillna(0, inplace = True)

train.head(5)

train["id_31"][train["id_31"] != 0] = 1

test["id_31"].fillna(0, inplace = True)

train.head(5)

test["id_31"][test["id_31"] != 0] = 1

test["id_31"].head(10)

for col in train.columns:

    print(col)
one_value_cols = [col for col in train.columns if train[col].nunique() <= 1]

many_null_cols = [col for col in train.columns if train[col].isnull().sum() / train.shape[0] > 0.9]

big_top_value_cols = [col for col in train.columns if train[col].value_counts(dropna=False, normalize=True).values[0] > 0.9]

one_value_cols_test = [col for col in test.columns if test[col].nunique() <= 1]

man_null_cols_test = [col for col in test.columns if test[col].isnull().sum() / train.shape[0] > 0.9]

big_top_value_cols_test = [col for col in test.columns if test[col].value_counts(dropna = False, normalize = True).values[0] > 0.9]

col_drop = list(set(one_value_cols + many_null_cols + big_top_value_cols + one_value_cols_test + man_null_cols_test + big_top_value_cols_test ))

col_drop.remove('isFraud')

print('{} total number of droped features.'.format(len(col_drop)))


train.drop(col_drop , axis = 1)

test_col = ['V133','V129','V299', 'V106', 'V301', 'V318', 'V119', 'V296', 'V113', 'V110', 'V112', 'V300', 'V105', 'D7' ,'V108' ,'V320', 'V121', 'V111' ,'V132', 'V117', 'V134' ,'V290', 'V98', 'V101', 'V136', 'V124', 'V297', 'V311', 'V284' ,'V109', 'V122', 'V115' ,'V305' ,'V137', 'V321', 'V281', 'V309', 'V114', 'V123', 'V319', 'V293', 'dist2', 'V125', 'V135', 'V286', 'V298', 'V316', 'V104', 'V103', 'V120' ,'V116', 'V118', 'V295', 'V107', 'C3', 'V102']

test_col[2]

for i in test_col:

    print(i)

    col_drop.remove(i)

    

print(len(col_drop))

test.drop(col_drop , axis = 1)
m = 0

for col in train.columns: 

    if train[col].dtype == 'object':

        le = LabelEncoder()

        le.fit(list(train[col].astype(str).values) )

        train[col] = le.transform(list(train[col].astype(str).values))

        m = m + 1

print(m)

del m



n = 0

for col in test.columns: 

    if test[col].dtype == 'object':

        le = LabelEncoder()

        le.fit(list(test[col].astype(str).values) )

        test[col] = le.transform(list(test[col].astype(str).values))

        n = n + 1

print(n)

del n

for c in train.columns:

    if train[c].dtype=='float16' or  train[c].dtype=='float32' or  train[c].dtype=='float64':

        train[c].fillna(train[c].mean())



for c in test.columns:

    if test[c].dtype=='float16' or  test[c].dtype=='float32' or  test[c].dtype=='float64':

        test[c].fillna(test[c].mean())        

        

y = train.iloc[:, 0].values

train = train.drop('isFraud', axis = 1)

x = train.iloc[:,:].values

#test = test.iloc[:,:].values
from sklearn.model_selection import StratifiedKFold

kfold = 5

skf = StratifiedKFold(n_splits=kfold, random_state=42)
params = {

    'min_child_weight': 10.0,

    'objective': 'binary:logistic',

    'max_depth': 7,

    'max_delta_step': 1.8,

    'colsample_bytree': 0.4,

    'subsample': 0.8,

    'eta': 0.025,

    'gamma': 0.65,

    'num_boost_round' : 700

    }
def gini(actual, pred, cmpcol = 0, sortcol = 1):

    assert( len(actual) == len(pred) )

    all = np.asarray(np.c_[ actual, pred, np.arange(len(actual)) ], dtype=np.float)

    all = all[ np.lexsort((all[:,2], -1*all[:,1])) ]

    totalLosses = all[:,0].sum()

    giniSum = all[:,0].cumsum().sum() / totalLosses

    

    giniSum -= (len(actual) + 1) / 2.

    return giniSum / len(actual)

 

def gini_normalized(a, p):

    return gini(a, p) / gini(a, a)



def gini_xgb(preds, dtrain):

    labels = dtrain.get_label()

    gini_score = gini_normalized(labels, preds)

    return 'gini', gini_score

import xgboost as xgb

for i, (train_index, test_index) in enumerate(skf.split(x, y)):

    print('[Fold %d/%d]' % (i + 1, kfold))

    X_train, X_valid = x[train_index], x[test_index]

    y_train, y_valid = y[train_index], y[test_index]

    # Convert our data into XGBoost format

    d_train = xgb.DMatrix(X_train, y_train)

    d_valid = xgb.DMatrix(X_valid, y_valid)

    d_test = xgb.DMatrix(test.values)

    watchlist = [(d_train, 'train'), (d_valid, 'valid')]



    # Train the model! We pass in a max of 1,600 rounds (with early stopping after 70)

    # and the custom metric (maximize=True tells xgb that higher metric is better)

    mdl = xgb.train(params, d_train, 1600, watchlist, early_stopping_rounds=70, feval=gini_xgb, maximize=True, verbose_eval=100)



    print('[Fold %d/%d Prediciton:]' % (i + 1, kfold))

    # Predict on our test data

    p_test = mdl.predict(d_test, ntree_limit=mdl.best_ntree_limit)

    sub['target'] += p_test/kfold