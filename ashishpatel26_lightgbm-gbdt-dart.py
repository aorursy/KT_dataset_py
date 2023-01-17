# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.metrics import roc_auc_score



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.metrics import mean_squared_error

import gc

import time

from pandas.core.common import SettingWithCopyWarning

import lightgbm as lgb

from xgboost import XGBRegressor

from catboost import CatBoostRegressor, Pool

from sklearn.model_selection import GroupKFold, GridSearchCV

from imblearn.over_sampling import SMOTE

from sklearn.model_selection import KFold, StratifiedKFold

from sklearn.metrics import roc_auc_score

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler



# I don't like SettingWithCopyWarnings ...

import warnings

warnings.filterwarnings('ignore')



%matplotlib inline

plt.style.use("fivethirtyeight")
# Display/plot feature importance

def display_importances(feature_importance_df_):

    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:40].index

    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]



    plt.figure(figsize=(8, 10))

    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))

    plt.title('LightGBM Features (avg over folds)')

    plt.tight_layout()

    plt.savefig('lgbm_importances.png')

    

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
def load_data():

    train = pd.read_csv('../input/train.csv', low_memory=True)

    test = pd.read_csv('../input/test.csv', low_memory=True)

    return train,test

train, test = load_data()

print("Train Shape:", train.shape)

print("Test Shape:",test.shape)

gc.enable()
train.head()
train['target'].value_counts().plot(kind="barh", figsize=(20,8))

for i, v in enumerate(train['target'].value_counts()):

    plt.text(v, i, str(v), fontweight='bold', fontsize = 20)

plt.xlabel("Count", fontsize=12)

plt.ylabel("State of the target", fontsize=12)

plt.title("Target repartition", fontsize=15)

plt.legend()

plt.show()



y = train.pop('target')
train.pop('ID_code')

test.pop('ID_code')

tr_col = train.columns
# train_df,y = SMOTE().fit_resample(train,y.ravel())
# train_df = pd.DataFrame(train_df)

# train_df.columns = tr_col

# train_df.head()
# y = pd.Series(y)
train_df = train.copy(deep=True)

print("Train Shape:", train_df.shape)

print("Target Shape:", y.shape)

gc.collect()
# y.value_counts().plot(kind="barh", figsize=(20,8))

# for i, v in enumerate(y.value_counts()):

#     plt.text(v, i, str(v), fontweight='bold', fontsize = 20)

# plt.xlabel("Count", fontsize=12)

# plt.ylabel("State of the target", fontsize=12)

# plt.title("Target repartition", fontsize=15)

# plt.legend()

# plt.show()
test_df = test.copy(deep=True)

print("Train Shape:", train_df.shape)

print("Target Shape:", y.shape)

print("Test Shape:", test_df.shape)
train_df = StandardScaler().fit_transform(train_df)

test_df = StandardScaler().fit_transform(test_df)

train_df.shape,test_df.shape
train_df = pd.DataFrame(train_df)

train_df.columns = tr_col
boosting = ["goss","dart"]

def kfold_lightgbm(train_df, test_df, num_folds, stratified = False, boosting = boosting[0]):

    print("Starting LightGBM. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))



    # Cross validation model

    if stratified:

        folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=326)

    else:

        folds = KFold(n_splits= num_folds, shuffle=True, random_state=2045)



    # Create arrays and dataframes to store results

    oof_preds = np.zeros(train_df.shape[0])

    sub_preds = np.zeros(test_df.shape[0])

    feature_importance_df = pd.DataFrame()

    

    # k-fold

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df, y)):

        train_x, train_y = train_df.iloc[train_idx], y.iloc[train_idx]

        valid_x, valid_y = train_df.iloc[valid_idx], y.iloc[valid_idx]



        # set data structure

        lgb_train = lgb.Dataset(train_x,label=train_y,free_raw_data=False)

        lgb_test = lgb.Dataset(valid_x,label=valid_y,free_raw_data=False)



        # params optimized by optuna

        params ={

                        'task': 'train',

                        'boosting': 'goss',

                        'objective': 'binary',

                        'metric': 'auc',

                        'learning_rate': 0.01,

                        'subsample': 0.8,

                        'max_depth': -1,

                        'top_rate': 0.9064148448434349,

                        'num_leaves': 32,

                        'min_child_weight': 41.9612869171337,

                        'other_rate': 0.0721768246018207,

                        'reg_alpha': 9.677537745007898,

                        'colsample_bytree': 0.5665320670155495,

                        'min_split_gain': 9.820197773625843,

                        'reg_lambda': 8.2532317400459,

                        'min_data_in_leaf': 21,

                        'verbose': -1,

                        'seed':int(2**n_fold),

                        'bagging_seed':int(2**n_fold),

                        'drop_seed':int(2**n_fold)

                        }



        reg = lgb.train(

                        params,

                        lgb_train,

                        valid_sets=[lgb_train, lgb_test],

                        valid_names=['train', 'test'],

                        num_boost_round=7000,early_stopping_rounds= 200,

                        verbose_eval=100,

                        )



        oof_preds[valid_idx] = reg.predict(valid_x, num_iteration=reg.best_iteration)

        sub_preds += reg.predict(test_df, num_iteration=reg.best_iteration) / folds.n_splits



        fold_importance_df = pd.DataFrame()

        fold_importance_df["feature"] = train_x.columns

        fold_importance_df["importance"] = np.log1p(reg.feature_importance(importance_type='gain', iteration=reg.best_iteration))

        fold_importance_df["fold"] = n_fold + 1

        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

        print('Fold %2d roc_auc_score : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx])))

        del reg, train_x, train_y, valid_x, valid_y

        gc.collect()



    # display importances

    display_importances(feature_importance_df)

    

        # save submission file

    submission = pd.read_csv("../input/sample_submission.csv")

    submission['target'] = sub_preds

    submission.to_csv(boosting+".csv", index=False)

    display(submission.head())

    return (submission)
submission = kfold_lightgbm(train_df, test_df, num_folds=5, stratified=True, boosting=boosting[0])   
submission1 = kfold_lightgbm(train_df, test_df, num_folds=5, stratified=True, boosting=boosting[1])   