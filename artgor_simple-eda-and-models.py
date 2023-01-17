import numpy as np

import pandas as pd

import os



import matplotlib.pyplot as plt

%matplotlib inline

from tqdm import tqdm_notebook

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_absolute_error

pd.options.display.precision = 15



import time

# Libraries

import numpy as np

import pandas as pd

pd.set_option('max_columns', None)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



import datetime

import lightgbm as lgb

from scipy import stats

from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, cross_val_score, GridSearchCV, RepeatedStratifiedKFold

from sklearn.preprocessing import StandardScaler

import os

import lightgbm as lgb

from sklearn.metrics import accuracy_score, roc_auc_score

from sklearn import metrics

from sklearn import linear_model

from tqdm import tqdm_notebook

from catboost import CatBoostClassifier



sensor = pd.read_csv('../input/sensor.csv')

sensor.drop(['Unnamed: 0'], axis=1, inplace=True)
sensor.head()
sensor.drop(['sensor_15'], axis=1, inplace=True)
plt.plot(sensor.loc[sensor['machine_status'] == 'NORMAL', 'sensor_02'], label='NORMAL')

plt.plot(sensor.loc[sensor['machine_status'] == 'BROKEN', 'sensor_02'], label='BROKEN')

plt.plot(sensor.loc[sensor['machine_status'] == 'RECOVERING', 'sensor_02'], label='RECOVERING')

plt.legend()
sensor['target'] = 0

sensor.loc[sensor['machine_status'] != 'NORMAL', 'target'] = 1
sensor.drop(['machine_status'], axis=1, inplace=True)
for col in sensor.columns[1:-1]:

    sensor[col] = sensor[col].fillna(sensor[col].mean())
X = sensor.drop(['timestamp', 'target'], axis=1)

y = sensor['target']

n_fold = 5

folds = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=42)
def train_model(X, y, params, folds, model_type='lgb', plot_feature_importance=False, averaging='usual', model=None):

    oof = np.zeros(len(X))

    scores = []

    feature_importance = pd.DataFrame()

    for fold_n, (train_index, valid_index) in enumerate(folds.split(X, y)):

        print('Fold', fold_n, 'started at', time.ctime())

        X_train, X_valid = X.loc[train_index], X.loc[valid_index]

        y_train, y_valid = y[train_index], y[valid_index]

        scaler = StandardScaler()

        X_train = scaler.fit_transform(X_train)

        X_valid = scaler.transform(X_valid)

        

        if model_type == 'lgb':

            train_data = lgb.Dataset(X_train, label=y_train)

            valid_data = lgb.Dataset(X_valid, label=y_valid)

            

            model = lgb.train(params,

                    train_data,

                    num_boost_round=1000,

                    valid_sets = [train_data, valid_data],

                    verbose_eval=500,

                    early_stopping_rounds = 200)

            

            y_pred_valid = model.predict(X_valid)

            

        if model_type == 'xgb':

            train_data = xgb.DMatrix(data=X_train, label=y_train, feature_names=X_train.columns)

            valid_data = xgb.DMatrix(data=X_valid, label=y_valid, feature_names=X_train.columns)



            watchlist = [(train_data, 'train'), (valid_data, 'valid_data')]

            model = xgb.train(dtrain=train_data, num_boost_round=20000, evals=watchlist, early_stopping_rounds=200, verbose_eval=500, params=params)

            y_pred_valid = model.predict(xgb.DMatrix(X_valid, feature_names=X_train.columns), ntree_limit=model.best_ntree_limit)

        

        if model_type == 'sklearn':

            model = model

            model.fit(X_train, y_train)

            y_pred_valid = model.predict_proba(X_valid)[:, 1]

            score = roc_auc_score(y_valid, y_pred_valid)

            # print(f'Fold {fold_n}. AUC: {score:.4f}.')

            # print('')

            

            

        if model_type == 'glm':

            model = sm.GLM(y_train, X_train, family=sm.families.Binomial())

            model_results = model.fit()

            model_results.predict(X_test)

            y_pred_valid = model_results.predict(X_valid).reshape(-1,)

            score = roc_auc_score(y_valid, y_pred_valid)

            

            

        if model_type == 'cat':

            model = CatBoostClassifier(iterations=20000, learning_rate=0.05, loss_function='Logloss',  eval_metric='AUC', **params)

            model.fit(X_train, y_train, eval_set=(X_valid, y_valid), cat_features=[], use_best_model=True, verbose=False)



            y_pred_valid = model.predict_proba(X_valid)[:, 1]

            

        oof[valid_index] = y_pred_valid.reshape(-1,)

        scores.append(roc_auc_score(y_valid, y_pred_valid))

        

        if model_type == 'lgb':

            # feature importance

            fold_importance = pd.DataFrame()

            fold_importance["feature"] = X.columns

            fold_importance["importance"] = model.feature_importance()

            fold_importance["fold"] = fold_n + 1

            feature_importance = pd.concat([feature_importance, fold_importance], axis=0)





    

    print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))

    

    if model_type == 'lgb':

        feature_importance["importance"] /= n_fold

        if plot_feature_importance:

            cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(

                by="importance", ascending=False)[:50].index



            best_features = feature_importance.loc[feature_importance.feature.isin(cols)]



            plt.figure(figsize=(16, 12));

            sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False));

            plt.title('LGB Features (avg over folds)');

        

            return oof,  feature_importance

        return oof,  scores

    

    else:

        return oof,  scores
params = {'num_leaves': 8,

         'min_data_in_leaf': 42,

         'objective': 'binary',

         'max_depth': 5,

         'learning_rate': 0.01,

         'boosting': 'gbdt',

         'bagging_freq': 5,

         'feature_fraction': 0.8201,

         'bagging_seed': 11,

         'reg_alpha': 1,

         'reg_lambda': 4,

         'random_state': 42,

         'metric': 'auc',

         'verbosity': -1,

         'subsample': 0.81,

         'num_threads': 4}

oof_lgb, scores = train_model(X, y, params=params, folds=folds, model_type='lgb', plot_feature_importance=True)
%%time

model = linear_model.LogisticRegression(class_weight='balanced', penalty='l2', C=0.1)

oof_lr, scores = train_model(X, y, params=None, folds=folds, model_type='sklearn', model=model)