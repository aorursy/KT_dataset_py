import pandas as pd

import numpy as np

from datetime import datetime

import matplotlib.pyplot as plt

from sklearn.model_selection import KFold

from sklearn.feature_selection import RFECV

import lightgbm as lgb

from tqdm import tqdm_notebook

from sklearn.metrics import roc_auc_score

from sklearn.preprocessing import LabelEncoder

import multiprocessing

import gc
files = ['../input/test_identity.csv', 

         '../input/test_transaction.csv',

         '../input/train_identity.csv',

         '../input/train_transaction.csv',

         '../input/sample_submission.csv']



def load_data(file):

    return pd.read_csv(file)



with multiprocessing.Pool() as pool:

    test_id, test_tr, train_id, train_tr, sub = pool.map(load_data, files)
train = pd.merge(train_tr, train_id, on='TransactionID', how='left')

test = pd.merge(test_tr, test_id, on='TransactionID', how='left')



del test_id, test_tr, train_id, train_tr

gc.collect()



one_value_cols = [col for col in train.columns if train[col].nunique() <= 1]

one_value_cols_test = [col for col in test.columns if test[col].nunique() <= 1]



many_null_cols = [col for col in train.columns if train[col].isnull().sum() / train.shape[0] > 0.9]

many_null_cols_test = [col for col in test.columns if test[col].isnull().sum() / test.shape[0] > 0.9]



big_top_value_cols = [col for col in train.columns if train[col].value_counts(dropna=False, normalize=True).values[0] > 0.9]

big_top_value_cols_test = [col for col in test.columns if test[col].value_counts(dropna=False, normalize=True).values[0] > 0.9]



cols_to_drop = list(set(many_null_cols + many_null_cols_test + big_top_value_cols + big_top_value_cols_test + one_value_cols + one_value_cols_test))

cols_to_drop.remove('isFraud')

print('{} features are going to be dropped for being useless'.format(len(cols_to_drop)))



train = train.drop(cols_to_drop, axis=1)

test = test.drop(cols_to_drop, axis=1)
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

    if verbose: 

        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df
for col in tqdm_notebook(train.columns): 

    if train[col].dtype == 'object':

        le = LabelEncoder()

        le.fit(list(train[col].astype(str).values) + list(test[col].astype(str).values))

        train[col] = le.transform(list(train[col].astype(str).values))
del test

gc.collect()
train = reduce_mem_usage(train)
X = train.sort_values('TransactionDT').drop(['isFraud', 'TransactionDT', 'TransactionID'], axis=1)

y = train.sort_values('TransactionDT')['isFraud']



del train

gc.collect()
# RFECV does not support NaNs

X.fillna(-999, inplace=True)
params = {'num_leaves': 491,

          'min_child_weight': 0.03454472573214212,

          'feature_fraction': 0.3797454081646243,

          'bagging_fraction': 0.4181193142567742,

          'min_data_in_leaf': 106,

          'objective': 'binary',

          'max_depth': -1,

          'learning_rate': 0.006883242363721497,

          "boosting_type": "gbdt",

          "bagging_seed": 11,

          "metric": 'auc',

          "verbosity": -1,

          'reg_alpha': 0.3899927210061127,

          'reg_lambda': 0.6485237330340494,

          'random_state': 47

         }
clf = lgb.LGBMClassifier(**params)

rfe = RFECV(estimator=clf, step=10, cv=KFold(n_splits=5, shuffle=False), scoring='roc_auc', verbose=2)
rfe.fit(X, y)
print('Optimal number of features:', rfe.n_features_)
plt.figure(figsize=(14, 8))

plt.xlabel("Number of features selected")

plt.ylabel("Cross validation score")

plt.plot(range(1, len(rfe.grid_scores_) + 1), rfe.grid_scores_)

plt.show()
for col in X.columns[rfe.ranking_ == 1]:

    print(col)