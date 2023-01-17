# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from tqdm import tqdm

import os

import pandas as pd

from scipy.stats import skew, kurtosis

import datetime

import sys

from sklearn.preprocessing import LabelEncoder



sys.path.append('.')

tqdm.pandas(ncols=50)

df_train = pd.read_csv('../input/loldataset/train.csv')  # (146022, 2)

df_test = pd.read_csv('../input/loldataset/test.csv')  # (36506, 1)

matches = pd.read_csv('../input/loldataset/matches.csv')  # (182528, 8)

# champs = pd.read_csv('data/champs.csv')

df_train = df_train.merge(matches, on='id', how='inner')

df_test = df_test.merge(matches, on='id', how='inner')

print(df_train.shape)

role = pd.read_csv('../input/loldataset/role.csv')

role['id'] = role['matchid']

del role['matchid']



train_sise = len(df_train)

df = pd.concat([df_train, df_test], axis=0, sort=False).reset_index(drop=True)

print(df.shape)



lb = LabelEncoder()

df['platformid'] = lb.fit_transform(df['platformid'])

df['version'] = lb.fit_transform(df['version'])

# df['role'] = lb.fit_transform(df['role'])

# df['position'] = lb.fit_transform(df['position'])



# wardsbought,,role,position



cols = role.drop(['id', 'player', 'role', 'position', 'wardsbought'], axis=1).columns.tolist()

cols_new = ['{}_{}'.format(col, i) for col in cols for i in range(1, 11)]

print(len(cols_new),cols_new)

role_grouped_df = pd.DataFrame({'id': role['id'].unique()})

features = []

for index, group in tqdm(role.groupby(by='id')):

    group = group.drop(['id', 'player', 'role', 'position', 'wardsbought'], axis=1)

    features.append(group.values.reshape(1,-1).tolist()[0])

tmp = pd.DataFrame(features)

tmp.columns=cols_new



role_features = pd.concat([role_grouped_df, tmp], axis=1)

df = pd.merge(df, role_features, on='id', how='left')

train, test = df[:train_sise], df[train_sise:]

del role_features

#

no_features = ['id', 'label', 'gameid', 'wardsbought']

features = [fea for fea in train.columns if fea not in no_features]

print(len(features), features)

print(train.shape)

print(test.shape)

def load_data():

    return train, test, no_features, features

import sys

from sklearn.preprocessing import LabelEncoder



sys.path.append('.')



import time

import lightgbm as lgb

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns

import xgboost as xgb

from catboost import CatBoostClassifier

from sklearn import metrics

from sklearn.model_selection import StratifiedKFold

#from load_train_test import load_data



train, test, no_features, features = load_data()

def train_model_classification(X, X_test, y, params, num_classes=2,

                               folds=None, model_type='lgb',

                               eval_metric='logloss', columns=None,

                               plot_feature_importance=False,

                               model=None, verbose=10000,

                               early_stopping_rounds=200,

                               splits=None, n_folds=3):

    """

    分类模型函数

    返回字典，包括： oof predictions, test predictions, scores and, if necessary, feature importances.



    :params: X - 训练数据， pd.DataFrame

    :params: X_test - 测试数据，pd.DataFrame

    :params: y - 目标

    :params: folds - folds to split data

    :params: model_type - 模型

    :params: eval_metric - 评价指标

    :params: columns - 特征列

    :params: plot_feature_importance - 是否展示特征重要性

    :params: model - sklearn model, works only for "sklearn" model type



    """

    start_time = time.time()

    global y_pred_valid, y_pred



    columns = X.columns if columns is None else columns

    X_test = X_test[columns]

    splits = folds.split(X, y) if splits is None else splits

    n_splits = folds.n_splits if splits is None else n_folds



    # to set up scoring parameters

    metrics_dict = {

        'logloss': {

            'lgb_metric_name': 'logloss',

            'xgb_metric_name': 'mlogloss',

            'catboost_metric_name': 'Logloss',

            'sklearn_scoring_function': metrics.log_loss

        },

        'lb_score_method': {

            'sklearn_scoring_f1': metrics.f1_score,  # 线上评价指标

            'sklearn_scoring_accuracy': metrics.accuracy_score,  # 线上评价指标

            'sklearn_scoring_auc': metrics.roc_auc_score

        },

    }

    result_dict = {}



    # out-of-fold predictions on train data

    oof = np.zeros(shape=(len(X), num_classes))

    # averaged predictions on train data

    prediction = np.zeros(shape=(len(X_test), num_classes))

    # list of scores on folds

    scores = []

    # feature importance

    feature_importance = pd.DataFrame()



    # split and train on folds

    for fold_n, (train_index, valid_index) in enumerate(splits):

        if verbose:

            print(f'Fold {fold_n + 1} started at {time.ctime()}')

        if type(X) == np.ndarray:

            X_train, X_valid = X[train_index], X[valid_index]

            y_train, y_valid = y[train_index], y[valid_index]

        else:

            X_train, X_valid = X[columns].iloc[train_index], X[columns].iloc[valid_index]

            y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]



        if model_type == 'lgb':

            model = lgb.LGBMClassifier(**params)

            model.fit(X_train, y_train,

                      eval_set=[(X_train, y_train), (X_valid, y_valid)],

                      eval_metric=metrics_dict[eval_metric]['lgb_metric_name'],

                      verbose=verbose,

                      early_stopping_rounds=early_stopping_rounds)



            y_pred_valid = model.predict_proba(X_valid)

            y_pred = model.predict_proba(X_test, num_iteration=model.best_iteration_)



        if model_type == 'xgb':

            model = xgb.XGBClassifier(**params)

            model.fit(X_train, y_train,

                      eval_set=[(X_train, y_train), (X_valid, y_valid)],

                      eval_metric=metrics_dict[eval_metric]['xgb_metric_name'],

                      verbose=bool(verbose),  # xgb verbose bool

                      early_stopping_rounds=early_stopping_rounds)

            y_pred_valid = model.predict_proba(X_valid)

            y_pred = model.predict_proba(X_test, ntree_limit=model.best_ntree_limit)

        if model_type == 'sklearn':

            model = model

            model.fit(X_train, y_train)

            y_pred_valid = model.predict_proba(X_valid)

            score = metrics_dict[eval_metric]['sklearn_scoring_function'](y_valid, y_pred_valid)

            print(f'Fold {fold_n}. {eval_metric}: {score:.4f}.')

            y_pred = model.predict_proba(X_test)



        if model_type == 'cat':

            model = CatBoostClassifier(iterations=20000, eval_metric=metrics_dict[eval_metric]['catboost_metric_name'],

                                       **params,

                                       loss_function=metrics_dict[eval_metric]['catboost_metric_name'])

            model.fit(X_train, y_train, eval_set=(X_valid, y_valid), cat_features=[], use_best_model=True,

                      verbose=False)



            y_pred_valid = model.predict_proba(X_valid)

            y_pred = model.predict_proba(X_test)



        oof[valid_index] = y_pred_valid

        # 评价指标

        scores.append(metrics_dict['lb_score_method']['sklearn_scoring_f1'](y_valid, np.argmax(y_pred_valid, axis=1),

                                                                            average='macro'))

        prediction += y_pred



        if model_type == 'lgb' and plot_feature_importance:

            # feature importance

            fold_importance = pd.DataFrame()

            fold_importance["feature"] = columns

            fold_importance["importance"] = model.feature_importances_

            fold_importance["fold"] = fold_n + 1

            feature_importance = pd.concat([feature_importance, fold_importance], axis=0)



        if model_type == 'xgb' and plot_feature_importance:

            # feature importance

            fold_importance = pd.DataFrame()

            fold_importance["feature"] = columns

            fold_importance["importance"] = model.feature_importances_

            fold_importance["fold"] = fold_n + 1

            feature_importance = pd.concat([feature_importance, fold_importance], axis=0)

    prediction /= n_splits

    print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))



    result_dict['oof'] = oof

    result_dict['prediction'] = prediction

    result_dict['scores'] = scores



    # if model_type == 'lgb' or model_type == 'xgb':

    #     if plot_feature_importance:

    #         feature_importance["importance"] /= n_splits

    #         cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(

    #             by="importance", ascending=False)[:50].index

    #

    #         best_features = feature_importance.loc[feature_importance.feature.isin(cols)]

    #

    #         plt.figure(figsize=(16, 12))

    #         sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))

    #         plt.title('LGB Features (avg over folds)')

    #         plt.show()

    #         result_dict['feature_importance'] = feature_importance

    end_time = time.time()



    print("train_model_classification cost time:{}".format(end_time - start_time))

    return result_dict

lgb_params = {

    'boosting_type': 'gbdt',

    'objective': 'binary',

    'metric': 'auc',

    'n_estimators': 100000,

    'learning_rate': 0.1,

    'random_state': 2948,

    'bagging_freq': 8,

    'bagging_fraction': 0.80718,

    # 'bagging_seed': 11,

    'feature_fraction': 0.38691,  # 0.3

    'feature_fraction_seed': 11,

    'max_depth': 9,

    'min_data_in_leaf': 40,

    'min_child_weight': 0.18654,

    "min_split_gain": 0.35079,

    'min_sum_hessian_in_leaf': 1.11347,

    'num_leaves': 29,

    'num_threads': 4,

    "lambda_l1": 0.55831,

    'lambda_l2': 1.67906,

    'cat_smooth': 10.4,

    'subsample': 0.7,

    'colsample_bytree': 0.7,

    # 'verbosity': 1,

}

n_fold = 5

folds = StratifiedKFold(n_splits=n_fold, random_state=1314)



X = train[features]

X_test = test[features]

num_classes = 2

y = train['label']
result_dict_lgb = train_model_classification(X=X,

                                             X_test=X_test,

                                             y=y,

                                             params=lgb_params,

                                             num_classes=num_classes,

                                             folds=folds,

                                             model_type='lgb',

                                             eval_metric='logloss',

                                             plot_feature_importance=True,

                                             verbose=100,

                                             early_stopping_rounds=200)



pred = np.argmax(result_dict_lgb['prediction'], axis=1)

test['label'] = pred



test['label'].to_csv('testsubmit.csv', header=False)
