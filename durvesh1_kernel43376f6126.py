# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import preprocessing



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df_trans = pd.read_csv('../input/ieee-fraud-detection/train_transaction.csv')

df_test_trans = pd.read_csv('../input/ieee-fraud-detection/test_transaction.csv')



df_id = pd.read_csv('../input/ieee-fraud-detection/train_identity.csv')

df_test_id = pd.read_csv('../input/ieee-fraud-detection/test_identity.csv')



sample_submission = pd.read_csv('../input/ieee-fraud-detection/sample_submission.csv', index_col='TransactionID')



df_train = df_trans.merge(df_id, how='left', left_index=True, right_index=True, on='TransactionID')

df_test = df_test_trans.merge(df_test_id, how='left', left_index=True, right_index=True, on='TransactionID')



print(df_train.shape)

print(df_test.shape)



# y_train = df_train['isFraud'].copy()

del df_trans, df_id, df_test_trans, df_test_id
def reduce_mem_usage(df, verbose=True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024**2    

    

    # Removing the extra storage space taken for the columns in the dataframe

    for col in df.columns:

#         print(col)

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
df_train = reduce_mem_usage(df_train)

df_test = reduce_mem_usage(df_test)
emails = {'gmail': 'google', 'att.net': 'att', 'twc.com': 'spectrum', 

          'scranton.edu': 'other', 'optonline.net': 'other', 'hotmail.co.uk': 'microsoft',

          'comcast.net': 'other', 'yahoo.com.mx': 'yahoo', 'yahoo.fr': 'yahoo',

          'yahoo.es': 'yahoo', 'charter.net': 'spectrum', 'live.com': 'microsoft', 

          'aim.com': 'aol', 'hotmail.de': 'microsoft', 'centurylink.net': 'centurylink',

          'gmail.com': 'google', 'me.com': 'apple', 'earthlink.net': 'other', 'gmx.de': 'other',

          'web.de': 'other', 'cfl.rr.com': 'other', 'hotmail.com': 'microsoft', 

          'protonmail.com': 'other', 'hotmail.fr': 'microsoft', 'windstream.net': 'other', 

          'outlook.es': 'microsoft', 'yahoo.co.jp': 'yahoo', 'yahoo.de': 'yahoo',

          'servicios-ta.com': 'other', 'netzero.net': 'other', 'suddenlink.net': 'other',

          'roadrunner.com': 'other', 'sc.rr.com': 'other', 'live.fr': 'microsoft',

          'verizon.net': 'yahoo', 'msn.com': 'microsoft', 'q.com': 'centurylink', 

          'prodigy.net.mx': 'att', 'frontier.com': 'yahoo', 'anonymous.com': 'other', 

          'rocketmail.com': 'yahoo', 'sbcglobal.net': 'att', 'frontiernet.net': 'yahoo', 

          'ymail.com': 'yahoo', 'outlook.com': 'microsoft', 'mail.com': 'other', 

          'bellsouth.net': 'other', 'embarqmail.com': 'centurylink', 'cableone.net': 'other', 

          'hotmail.es': 'microsoft', 'mac.com': 'apple', 'yahoo.co.uk': 'yahoo', 'netzero.com': 'other', 

          'yahoo.com': 'yahoo', 'live.com.mx': 'microsoft', 'ptd.net': 'other', 'cox.net': 'other',

          'aol.com': 'aol', 'juno.com': 'other', 'icloud.com': 'apple'}



us_emails = ['gmail', 'net', 'edu']



# https://www.kaggle.com/c/ieee-fraud-detection/discussion/100499#latest-579654

for c in ['P_emaildomain', 'R_emaildomain']:

    df_train[c + '_bin'] = df_train[c].map(emails)

    df_test[c + '_bin'] = df_test[c].map(emails)

    

    df_train[c + '_suffix'] = df_train[c].map(lambda x: str(x).split('.')[-1])

    df_test[c + '_suffix'] = df_test[c].map(lambda x: str(x).split('.')[-1])

    

    df_train[c + '_suffix'] = df_train[c + '_suffix'].map(lambda x: x if str(x) not in us_emails else 'us')

    df_test[c + '_suffix'] = df_test[c + '_suffix'].map(lambda x: x if str(x) not in us_emails else 'us')
for f in df_train.drop('isFraud', axis=1).columns:

    if df_train[f].dtype=='object' or df_test[f].dtype=='object': 

        lbl = preprocessing.LabelEncoder()

        lbl.fit(list(df_train[f].values) + list(df_test[f].values))

        df_train[f] = lbl.transform(list(df_train[f].values))

        df_test[f] = lbl.transform(list(df_test[f].values))  
df_train['Trans_min_mean'] = df_train['TransactionAmt'] - df_train['TransactionAmt'].mean()

df_train['Trans_min_std'] = df_train['Trans_min_mean'] / df_train['TransactionAmt'].std()

df_test['Trans_min_mean'] = df_test['TransactionAmt'] - df_test['TransactionAmt'].mean()

df_test['Trans_min_std'] = df_test['Trans_min_mean'] / df_test['TransactionAmt'].std()
df_train['TransactionAmt_to_mean_card1'] = df_train['TransactionAmt'] / df_train.groupby(['card1'])['TransactionAmt'].transform('mean')

df_train['TransactionAmt_to_mean_card4'] = df_train['TransactionAmt'] / df_train.groupby(['card4'])['TransactionAmt'].transform('mean')

df_train['TransactionAmt_to_std_card1'] = df_train['TransactionAmt'] / df_train.groupby(['card1'])['TransactionAmt'].transform('std')

df_train['TransactionAmt_to_std_card4'] = df_train['TransactionAmt'] / df_train.groupby(['card4'])['TransactionAmt'].transform('std')



df_test['TransactionAmt_to_mean_card1'] = df_test['TransactionAmt'] / df_test.groupby(['card1'])['TransactionAmt'].transform('mean')

df_test['TransactionAmt_to_mean_card4'] = df_test['TransactionAmt'] / df_test.groupby(['card4'])['TransactionAmt'].transform('mean')

df_test['TransactionAmt_to_std_card1'] = df_test['TransactionAmt'] / df_test.groupby(['card1'])['TransactionAmt'].transform('std')

df_test['TransactionAmt_to_std_card4'] = df_test['TransactionAmt'] / df_test.groupby(['card4'])['TransactionAmt'].transform('std')
df_train['TransactionAmt'] = np.log(df_train['TransactionAmt'])

df_test['TransactionAmt'] = np.log(df_test['TransactionAmt'])
df_test['isFraud'] = 'test'

df = pd.concat([df_train, df_test], axis=0, sort=False )

df = df.reset_index()

df = df.drop('index', axis=1)
def PCA_change(df, cols, n_components, prefix='PCA_', rand_seed=4):

    pca = PCA(n_components=n_components, random_state=rand_seed)



    principalComponents = pca.fit_transform(df[cols])



    principalDf = pd.DataFrame(principalComponents)



    df.drop(cols, axis=1, inplace=True)



    principalDf.rename(columns=lambda x: str(prefix)+str(x), inplace=True)



    df = pd.concat([df, principalDf], axis=1)

    

    return df
mas_v = df_train.columns[55:394]
from sklearn.preprocessing import minmax_scale

from sklearn.decomposition import PCA



for col in mas_v:

    df[col] = df[col].fillna((df[col].min() - 2))

    df[col] = (minmax_scale(df[col], feature_range=(0,1)))



    

df = PCA_change(df, mas_v, prefix='PCA_V_', n_components=30)
df = reduce_mem_usage(df)
df_train, df_test = df[df['isFraud'] != 'test'], df[df['isFraud'] == 'test'].drop('isFraud', axis=1)
X_train = df_train.sort_values('TransactionDT').drop(['isFraud', 'TransactionDT'],axis=1)

y_train = df_train.sort_values('TransactionDT')['isFraud'].astype(bool)



X_test = df_test.sort_values('TransactionDT').drop(['TransactionDT'],axis=1)

del df_train

df_test = df_test[["TransactionDT"]]
import numpy as np 

import pandas as pd

import scipy as sp

from scipy import stats

import matplotlib.pyplot as plt

import seaborn as sns





from sklearn import preprocessing

from sklearn.metrics import confusion_matrix, roc_auc_score

from sklearn.model_selection import StratifiedKFold, cross_val_score, KFold

from xgboost import XGBClassifier

import xgboost as xgb



## Hyperopt modules

from hyperopt import fmin, hp, tpe, Trials, space_eval, STATUS_OK, STATUS_RUNNING

from functools import partial
from sklearn.model_selection import KFold,TimeSeriesSplit

from sklearn.metrics import roc_auc_score

from xgboost import plot_importance

from sklearn.metrics import make_scorer

from hyperopt import fmin, hp, tpe

import gc



import time

def objective(params):

    time1 = time.time()

    params = {

        'max_depth': int(params['max_depth']),

        'gamma': "{:.3f}".format(params['gamma']),

        'subsample': "{:.2f}".format(params['subsample']),

        'reg_alpha': "{:.3f}".format(params['reg_alpha']),

        'reg_lambda': "{:.3f}".format(params['reg_lambda']),

        'learning_rate': "{:.3f}".format(params['learning_rate']),

        'num_leaves': '{:.3f}'.format(params['num_leaves']),

        'colsample_bytree': '{:.3f}'.format(params['colsample_bytree']),

        'min_child_samples': '{:.3f}'.format(params['min_child_samples']),

        'feature_fraction': '{:.3f}'.format(params['feature_fraction']),

        'bagging_fraction': '{:.3f}'.format(params['bagging_fraction'])

    }



    print("\n############## New Run ################")

    print(f"params = {params}")

    FOLDS = 7

    count=1

    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)



    tss = TimeSeriesSplit(n_splits=FOLDS)

    y_preds = np.zeros(sample_submission.shape[0])

    y_oof = np.zeros(X_train.shape[0])

    score_mean = 0

    for tr_idx, val_idx in tss.split(X_train, y_train):

        clf = xgb.XGBClassifier(

            n_estimators=600, random_state=4, verbose=True, 

            tree_method='gpu_hist', 

            **params

        )



        X_tr, X_vl = X_train.iloc[tr_idx, :], X_train.iloc[val_idx, :]

        y_tr, y_vl = y_train.iloc[tr_idx], y_train.iloc[val_idx]

        

        clf.fit(X_tr, y_tr)

        #y_pred_train = clf.predict_proba(X_vl)[:,1]

        #print(y_pred_train)

        score = make_scorer(roc_auc_score, needs_proba=True)(clf, X_vl, y_vl)

        # plt.show()

        score_mean += score

        print(f'{count} CV - score: {round(score, 4)}')

        count += 1

    time2 = time.time() - time1

    print(f"Total Time Run: {round(time2 / 60,2)}")

    gc.collect()

    print(f'Mean ROC_AUC: {score_mean / FOLDS}')

    fig, ax = plt.subplots(figsize=(20,20))

    plot_importance(clf, ax=ax)

    plt.show()

    del X_tr, X_vl, y_tr, y_vl, score, clf

    return (score_mean / FOLDS)





space={

    'max_depth': hp.quniform('max_depth', 7, 23, 1),

    'reg_alpha':  hp.uniform('reg_alpha', 0.01, 0.4),

    'reg_lambda': hp.uniform('reg_lambda', 0.01, .4),

    'learning_rate': hp.uniform('learning_rate', 0.01, 0.2),

    'colsample_bytree': hp.uniform('colsample_bytree', 0.3, .9),

    'gamma': hp.uniform('gamma', 0.01, .7),

    'num_leaves': hp.choice('num_leaves', list(range(20, 250, 10))),

    'min_child_samples': hp.choice('min_child_samples', list(range(100, 250, 10))),

    'subsample': hp.choice('subsample', [0.2, 0.4, 0.5, 0.6, 0.7, .8, .9]),

    'feature_fraction': hp.uniform('feature_fraction', 0.4, .8),

    'bagging_fraction': hp.uniform('bagging_fraction', 0.4, .9) 

    }
best = fmin(fn=objective,

            space=space,

            algo=tpe.suggest,

            max_evals=10)



# Print best parameters

best_params = space_eval(space, best)

print(best_params)