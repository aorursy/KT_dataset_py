# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
folder_path = '../input/ieee-fraud-detection/'

train_identity = pd.read_csv(folder_path + 'train_identity.csv')
test_identity = pd.read_csv(folder_path + 'test_identity.csv')

train_transaction = pd.read_csv(folder_path + 'train_transaction.csv')

test_transaction = pd.read_csv(folder_path + 'test_transaction.csv')    

sub = pd.read_csv(folder_path + 'sample_submission.csv')
import pandas as pd

from tqdm import tqdm_notebook

from sklearn.preprocessing import StandardScaler

from sklearn.svm import NuSVR, SVR

from sklearn.metrics import mean_absolute_error

pd.options.display.precision = 15



import lightgbm as lgb

import xgboost as xgb

import time

import datetime

# from catboost import CatBoostRegressor

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold, GroupKFold, GridSearchCV, train_test_split, TimeSeriesSplit

from sklearn import metrics

from sklearn import linear_model

import gc

import seaborn as sns

from sklearn import metrics

from itertools import product

import numpy as np

import matplotlib.pyplot as plt
train = pd.merge(train_transaction,train_identity,on = 'TransactionID', how = 'left')

test = pd.merge(test_transaction,test_identity,on='TransactionID',how='left')
del train_identity,train_transaction,test_identity,test_transaction
one_value_cols_test = [col for col in test.columns if test[col].nunique() <= 1]

one_value_cols_train = [col for col in train.columns if train[col].nunique() <= 1]
many_null_cols = [col for col in train.columns if train[col].isnull().sum() / train.shape[0] > 0.9]

many_null_cols_test = [col for col in test.columns if test[col].isnull().sum() / test.shape[0] > 0.9]

big_top_value_cols = [col for col in train.columns if train[col].value_counts(dropna=False, normalize=True).values[0] > 0.9]

big_top_value_cols_test = [col for col in test.columns if test[col].value_counts(dropna=False, normalize=True).values[0] > 0.9]

cols_to_drop = list(set(many_null_cols + many_null_cols_test + big_top_value_cols + big_top_value_cols_test + one_value_cols_train+ one_value_cols_test))

cols_to_drop.remove('isFraud')

len(cols_to_drop)
train = train.drop(cols_to_drop, axis=1)

test = test.drop(cols_to_drop, axis=1)
cols = ['isFraud','TransactionID','TransactionDT','card1','TransactionAmt','card2','addr1','D15','dist1','D4','D2','D13','D10','P_emaildomain','card5','D11','id_02','D8','D1','id_20','id_19','id_31','C1','C2','D5','D3','DeviceInfo','D9','C11','C6','id_06','V307','id_05','C14','C9','V313','V310','R_emaildomain','id_13','id_01','M4','id_33','C5','D14','card4','M6','M5','V314','V130','id_30','V317','V315','card6']

cols_1 = ['TransactionID','TransactionDT','card1','TransactionAmt','card2','addr1','D15','dist1','D4','D2','D13','D10','P_emaildomain','card5','D11','id_02','D8','D1','id_20','id_19','id_31','C1','C2','D5','D3','DeviceInfo','D9','C11','C6','id_06','V307','id_05','C14','C9','V313','V310','R_emaildomain','id_13','id_01','M4','id_33','C5','D14','card4','M6','M5','V314','V130','id_30','V317','V315','card6']
new_train = train[cols].copy()

new_test = test[cols_1].copy()
cat_cols = ['id_13','id_20', 'id_19', 'id_30', 'id_31', 'id_33','DeviceInfo', 'card4', 'card6', 'M4','P_emaildomain',

            'R_emaildomain', 'card1', 'card2', 'card5', 'M5', 'M6']
categotical_columns = ['id_13','id_30','card4', 'card6', 'M4','P_emaildomain','R_emaildomain', 'M5', 'M6']
for col in cat_cols:

    if col in train.columns:

        le = LabelEncoder()

        le.fit(list(new_train[col].astype(str).values) + list(new_test[col].astype(str).values))

        new_train[col] = le.transform(list(new_train[col].astype(str).values))

        new_test[col] = le.transform(list(new_test[col].astype(str).values)) 
X = new_train.sort_values('TransactionDT').drop(['isFraud','TransactionID','TransactionDT'],axis = 1)

y = new_train.sort_values('TransactionID')['isFraud']

X_test = new_test.drop(['TransactionID','TransactionDT'],axis = 1)
del train

new_test = new_test[["TransactionDT", 'TransactionID']]

del new_train
# by https://www.kaggle.com/dimartinot

def clean_inf_nan(df):

    return df.replace([np.inf, -np.inf], np.nan)   



# Cleaning infinite values to NaN

X = clean_inf_nan(X)

X_test = clean_inf_nan(X_test )
n_fold = 5

fold = StratifiedKFold(n_splits = 5)

params = {'num_leaves': 256,

          'min_child_samples': 79,

          'objective': 'binary',

          'max_depth': 13,

          'learning_rate': 0.03,

          "boosting_type": "gbdt",

          "subsample_freq": 3,

          "subsample": 0.9,

          "bagging_seed": 11,

          "metric": 'auc',

          "verbosity": -1,

          'reg_alpha': 0.3,

          'reg_lambda': 0.3,

          'colsample_bytree': 0.9,

#           'categorical_feature': categotical_columns

         }
for col in categotical_columns:

    X[col] = X[col].astype('category').copy()

    X_test[col] = X_test[col].astype('category').copy()

    
def fast_auc(y_true, y_prob):

    """

    fast roc_auc computation: https://www.kaggle.com/c/microsoft-malware-prediction/discussion/76013

    """

    y_true = np.asarray(y_true)

    y_true = y_true[np.argsort(y_prob)]

    nfalse = 0

    auc = 0

    n = len(y_true)

    for i in range(n):

        y_i = y_true[i]

        nfalse += (1 - y_i)

        auc += y_i * nfalse

    auc /= (nfalse * (n - nfalse))

    return auc





def eval_auc(y_true, y_pred):

    """

    Fast auc eval function for lgb.

    """

    return 'auc', fast_auc(y_true, y_pred), True



def train_model_classification(X, X_test, y, params, folds, model_type='lgb', eval_metric='auc', columns=None, plot_feature_importance=False, model=None,

                               verbose=10000, early_stopping_rounds=200, n_estimators=50000):

    """

    A function to train a variety of regression models.

    Returns dictionary with oof predictions, test predictions, scores and, if necessary, feature importances.

    

    :params: X - training data, can be pd.DataFrame or np.ndarray (after normalizing)

    :params: X_test - test data, can be pd.DataFrame or np.ndarray (after normalizing)

    :params: y - target

    :params: folds - folds to split data

    :params: model_type - type of model to use

    :params: eval_metric - metric to use

    :params: columns - columns to use. If None - use all columns

    :params: plot_feature_importance - whether to plot feature importance of LGB

    :params: model - sklearn model, works only for "sklearn" model type

    

    """

    columns = X.columns if columns == None else columns

    X_test = X_test[columns]

    

    # to set up scoring parameters

    metrics_dict = {'auc': {'lgb_metric_name': eval_auc,

                        'catboost_metric_name': 'AUC',

                        'sklearn_scoring_function': metrics.roc_auc_score},

                    }

    

    result_dict = {}

    

    # out-of-fold predictions on train data

    oof = np.zeros((len(X), len(set(y.values))))

    

    # averaged predictions on train data

    prediction = np.zeros((len(X_test), oof.shape[1]))

    

    # list of scores on folds

    scores = []

    feature_importance = pd.DataFrame()

    

    # split and train on folds

    for fold_n, (train_index, valid_index) in enumerate(folds.split(X,y)):

        print(f'Fold {fold_n + 1} started at {time.ctime()}')

        if type(X) == np.ndarray:

            X_train, X_valid = X[columns][train_index], X[columns][valid_index]

            y_train, y_valid = y[train_index], y[valid_index]

        else:

            X_train, X_valid = X[columns].iloc[train_index], X[columns].iloc[valid_index]

            y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

            

        if model_type == 'lgb':

            model = lgb.LGBMClassifier(**params, n_estimators=n_estimators, n_jobs = -1)

            model.fit(X_train, y_train, 

                    eval_set=[(X_train, y_train), (X_valid, y_valid)], eval_metric=metrics_dict[eval_metric]['lgb_metric_name'],

                    verbose=verbose, early_stopping_rounds=early_stopping_rounds)

            

            y_pred_valid = model.predict_proba(X_valid)

            y_pred = model.predict_proba(X_test, num_iteration=model.best_iteration_)

            

        if model_type == 'xgb':

            train_data = xgb.DMatrix(data=X_train, label=y_train, feature_names=X.columns)

            valid_data = xgb.DMatrix(data=X_valid, label=y_valid, feature_names=X.columns)



            watchlist = [(train_data, 'train'), (valid_data, 'valid_data')]

            model = xgb.train(dtrain=train_data, num_boost_round=n_estimators, evals=watchlist, early_stopping_rounds=early_stopping_rounds, verbose_eval=verbose, params=params)

            y_pred_valid = model.predict(xgb.DMatrix(X_valid, feature_names=X.columns), ntree_limit=model.best_ntree_limit)

            y_pred = model.predict(xgb.DMatrix(X_test, feature_names=X.columns), ntree_limit=model.best_ntree_limit)

        

        if model_type == 'sklearn':

            model = model

            model.fit(X_train, y_train)

            

            y_pred_valid = model.predict(X_valid).reshape(-1,)

            score = metrics_dict[eval_metric]['sklearn_scoring_function'](y_valid, y_pred_valid)

            print(f'Fold {fold_n}. {eval_metric}: {score:.4f}.')

            print('')

            

            y_pred = model.predict_proba(X_test)

        

        if model_type == 'cat':

            model = CatBoostClassifier(iterations=n_estimators, eval_metric=metrics_dict[eval_metric]['catboost_metric_name'], **params,

                                      loss_function=metrics_dict[eval_metric]['catboost_metric_name'])

            model.fit(X_train, y_train, eval_set=(X_valid, y_valid), cat_features=[], use_best_model=True, verbose=False)



            y_pred_valid = model.predict(X_valid)

            y_pred = model.predict(X_test)

        

        oof[valid_index] = y_pred_valid

        scores.append(metrics_dict[eval_metric]['sklearn_scoring_function'](y_valid, y_pred_valid[:, 1]))



        prediction += y_pred    

        

        if model_type == 'lgb' and plot_feature_importance:

            # feature importance

            fold_importance = pd.DataFrame()

            fold_importance["feature"] = columns

            fold_importance["importance"] = model.feature_importances_

            fold_importance["fold"] = fold_n + 1

            feature_importance = pd.concat([feature_importance, fold_importance], axis=0)



    prediction /= folds.n_splits

    

    print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))

    

    result_dict['oof'] = oof

    result_dict['prediction'] = prediction

    result_dict['scores'] = scores

    

    if model_type == 'lgb':

        if plot_feature_importance:

            feature_importance["importance"] /= folds.n_splits

            cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(

                by="importance", ascending=False)[:50].index



            best_features = feature_importance.loc[feature_importance.feature.isin(cols)]



            plt.figure(figsize=(16, 12));

            sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False));

            plt.title('LGB Features (avg over folds)');

            

            result_dict['feature_importance'] = feature_importance

        

    return result_dict
result_dict_lgb = train_model_classification(X=X, X_test=X_test, y=y, params=params, folds=fold, model_type='lgb', eval_metric='auc', plot_feature_importance=True,

                                                      verbose=500, early_stopping_rounds=200, n_estimators=5000)
sub['isFraud'] = result_dict_lgb['prediction']

sub.to_csv('submission.csv', index=False)