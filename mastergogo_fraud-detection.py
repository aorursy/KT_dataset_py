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



import numpy as np

import pandas as pd

from sklearn import preprocessing

import xgboost as xgb

import lightgbm as lgb

from sklearn.metrics import mean_squared_error
%%time

train_transaction = pd.read_csv('../input/train_transaction.csv', index_col='TransactionID')

test_transaction = pd.read_csv('../input/test_transaction.csv', index_col='TransactionID')



train_identity = pd.read_csv('../input/train_identity.csv', index_col='TransactionID')

test_identity = pd.read_csv('../input/test_identity.csv', index_col='TransactionID')



sample_submission = pd.read_csv('../input/sample_submission.csv', index_col='TransactionID')



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



# Label Encoding

for f in X_train.columns:

    if X_train[f].dtype=='object' or X_test[f].dtype=='object': 

        lbl = preprocessing.LabelEncoder()

        lbl.fit(list(X_train[f].values) + list(X_test[f].values))

        X_train[f] = lbl.transform(list(X_train[f].values))

        X_test[f] = lbl.transform(list(X_test[f].values))   
%%time

# From kernel https://www.kaggle.com/gemartin/load-data-reduce-memory-usage

# WARNING! THIS CAN DAMAGE THE DATA 

def reduce_mem_usage(df):

    """ iterate through all the columns of a dataframe and modify the data type

        to reduce memory usage.        

    """

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

        else:

            df[col] = df[col].astype('category')



    end_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))

    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    

    return df

X_train = reduce_mem_usage(X_train)

X_test = reduce_mem_usage(X_test)
cols_to_delete = []

for name, values in X_train.iteritems():

    if X_train[name].isnull().sum() > 500000:

        cols_to_delete.append(name)



X_train.drop(cols_to_delete,axis=1,inplace=True)

X_test.drop(cols_to_delete,axis=1,inplace=True)
X_train.isnull().sum()
# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

X_train_pre, X_test_pre, y_train_pre, y_test_pre = train_test_split(X_train, y_train, test_size = .10, random_state = 0,stratify=y_train)

lgb_train = lgb.Dataset(X_train_pre, y_train_pre)

lgb_eval = lgb.Dataset(X_test_pre, y_test_pre, reference=lgb_train)

params = {

    'application': 'binary',

    'objective': 'binary',

    'metric': 'auc',

    'boosting': 'gbdt',

    'num_leaves': 512,

    'feature_fraction': 0.5,

    'bagging_fraction': 0.80,

    'bagging_freq': 200,

    'learning_rate': 0.05,

    'min_data' : 500,

    'max_bin': 1024,

    'max_depth' : -1,

    'verbose': -1

}
print('Starting training...')

# train

gbm = lgb.train(params,

                lgb_train,

                num_boost_round=2000,

                valid_sets=lgb_eval,

                early_stopping_rounds=100)
sample_submission.head()
print('Starting predicting...')

# predict

y_preds = gbm.predict(X_test, num_iteration=gbm.best_iteration)

# eval/

#print('The rmse of prediction is:', mean_squared_error(y_test_pre, y_pred) ** 0.5)
sample_submission['isFraud'] = y_preds
sample_submission.to_csv('simple_lgbm.csv')