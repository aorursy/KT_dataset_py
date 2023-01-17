# General imports

import numpy as np

import pandas as pd

import os, sys, gc, warnings, random, datetime



from sklearn import metrics

from sklearn.model_selection import train_test_split, KFold, GroupKFold

from sklearn.preprocessing import LabelEncoder

import xgboost as xgb

from sklearn.metrics import roc_auc_score



from tqdm import tqdm



import math

warnings.filterwarnings('ignore')
########################### Helpers

#################################################################################

## Seeder

# :seed to make all processes deterministic     # type: int

def seed_everything(seed=0):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    

## Memory Reducer

# :df pandas dataframe to reduce size             # type: pd.DataFrame()

# :verbose                                        # type: bool

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
########################### Vars

#################################################################################

SEED = 42

seed_everything(SEED)

LOCAL_TEST = False

TARGET = 'isFraud'

START_DATE = datetime.datetime.strptime('2017-11-30', '%Y-%m-%d')
########################### DATA LOAD

#################################################################################

print('Load Data')





train_df = pd.read_pickle('../input/mydata/train_df_with_mn.pkl')

test_df = pd.read_pickle('../input/mydata/test_df_with_mn.pkl') 



# train_df = pd.read_pickle('/content/drive/My Drive/Colab Notebooks/ieee fraud detecting/train_df_jojim.pkl')

# test_df = pd.read_pickle('/content/drive/My Drive/Colab Notebooks/ieee fraud detecting/test_df_jojim.pkl')

    

remove_features = pd.read_pickle('../input/mydata/remove_features_with_mn.pkl')

#remove_features = pd.read_pickle('/content/drive/My Drive/Colab Notebooks/ieee fraud detecting/remove_features_jojim.pkl')
remove_features = list(remove_features['features_to_remove'].values)

remove_features.append("BrowserUpToDate")

print('Shape control:', train_df.shape, test_df.shape)
########################### Final features list

features_columns = [col for col in list(train_df) if col not in remove_features]



########################### Final Minification

## I don't like this part as it changes float numbers

## small change but change.

## To be able to train lgbm without 

## minification we need to do some changes on model

## we will do it later.

if not LOCAL_TEST:

    train_df = reduce_mem_usage(train_df)

    test_df  = reduce_mem_usage(test_df)
train_df[['P_emaildomain', 'R_emaildomain', 'DeviceType']] = train_df[['P_emaildomain', 'R_emaildomain', 'DeviceType']].astype(int)

test_df[['P_emaildomain', 'R_emaildomain', 'DeviceType']] = train_df[['P_emaildomain', 'R_emaildomain', 'DeviceType']].astype(int)
X_train = train_df[features_columns]

y_train = train_df[TARGET]

X_test = test_df[features_columns]
NFOLDS = 6

folds = GroupKFold(n_splits = NFOLDS)

y_preds = np.zeros(X_test.shape[0])

#y_oof = np.zeros(X_train.shape[0])

split_groups = train_df['DT_M']
del train_df, test_df
for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train, groups=split_groups)):



    clf = xgb.XGBClassifier(

        n_estimators=512,

        max_depth=16,

        learning_rate=0.014,

        subsample=0.85,

        colsample_bytree=0.85,

        tree_method='gpu_hist',

        reg_alpha=0.3,

        reg_lamdba=0.243

    )

    

    X_tr, X_vl = X_train.iloc[trn_idx, :], X_train.iloc[val_idx, :]

    y_tr, y_vl = y_train.iloc[trn_idx], y_train.iloc[val_idx]

    clf.fit(X_tr, y_tr)

    y_pred_train = clf.predict_proba(X_vl)[:,1]

    #y_oof[val_idx] = y_pred_train

    

    print('ROC AUC {}'.format(roc_auc_score(y_vl, y_pred_train)))

    

    del X_tr, X_vl, y_tr, y_vl

    

    y_preds[:int(len(X_test)/2)] += clf.predict_proba(X_test[:int(len(X_test)/2)])[:,1] / NFOLDS

    y_preds[int(len(X_test)/2):] += clf.predict_proba(X_test[int(len(X_test)/2):])[:,1] / NFOLDS



    gc.collect()
########################### Model Train

# if LOCAL_TEST:

#     lgb_params['learning_rate'] = 0.01

#     lgb_params['n_estimators'] = 10000

#     lgb_params['early_stopping_rounds'] = 100

#     test_predictions = make_predictions(train_df, test_df, features_columns, TARGET, lgb_params, NFOLDS=4)

# else:

#     lgb_params['learning_rate'] = 0.007

#     lgb_params['n_estimators'] = 10000

#     lgb_params['early_stopping_rounds'] = 100    

#     test_predictions = make_predictions(train_df, test_df, features_columns, TARGET, lgb_params, NFOLDS=6)
########################### Export

if not LOCAL_TEST:

    submission = pd.read_csv("../input/ieee-fraud-detection/sample_submission.csv")

    submission['isFraud'] = y_preds

    submission.to_csv('submission_xgb_groupkfold_final.csv', index=False)
submission