# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



import matplotlib.pyplot as plt

%matplotlib inline



import lightgbm as lgb

import time

import datetime



import json

import ast

import eli5

import shap

from eli5.sklearn import PermutationImportance

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold, train_test_split, GroupKFold, GroupShuffleSplit

from sklearn.metrics import accuracy_score, confusion_matrix

import gc

from catboost import CatBoostClassifier

import seaborn as sns

import warnings

warnings.filterwarnings("ignore")
sample_submission = pd.read_csv('../input/sample_submission.csv')

X_train = pd.read_csv('../input/X_train.csv')

y_train = pd.read_csv('../input/y_train.csv')

X_test = pd.read_csv('../input/X_test.csv')
sample_submission.head()
sample_submission.tail()
X_train.head()
X_train.tail()
X_train.columns
X_train.dtypes
X_train.info()
X_train.describe()
X_train.isnull().sum()
y_train.head()
y_train.tail()
y_train['surface'].unique()
X_test.head()
X_test.tail()
def aggregate(df):

    agg_func = {

        'orientation_X': ['sum','mean', 'max', 'min', 'std'],

        'orientation_Y': ['sum','mean', 'max', 'min', 'std'],

        'orientation_Z': ['sum','mean', 'max', 'min', 'std'],

        'orientation_W': ['sum','mean', 'max', 'min', 'std'],

        'angular_velocity_X': ['sum','mean', 'max', 'min', 'std'],

        'angular_velocity_Y': ['sum','mean', 'max', 'min', 'std'],

        'angular_velocity_Z': ['sum','mean', 'max', 'min', 'std'],

        'linear_acceleration_X': ['sum','mean', 'max', 'min', 'std'],

        'linear_acceleration_Y': ['sum','mean', 'max', 'min', 'std'],

        'linear_acceleration_Z': ['sum','mean', 'max', 'min', 'std'],

    }



    agg_df = df.groupby(['series_id']).agg(agg_func)

    agg_df.columns = ['_'.join(col).strip() for col in agg_df.columns.values]

    agg_df.reset_index(inplace=True)



    df = (df.groupby('series_id')

          .size()

          .reset_index(name='count'))



    agg_df = pd.merge(df, agg_df, on='series_id', how='left')



    agg_df

    return agg_df
def quaternion_to_euler(x, y, z, w):

    import math

    t0 = +2.0 * (w * x + y * z)

    t1 = +1.0 - 2.0 * (x * x + y * y)

    X = math.atan2(t0, t1)



    t2 = +2.0 * (w * y - z * x)

    t2 = +1.0 if t2 > +1.0 else t2

    t2 = -1.0 if t2 < -1.0 else t2

    Y = math.asin(t2)



    t3 = +2.0 * (w * z + x * y)

    t4 = +1.0 - 2.0 * (y * y + z * z)

    Z = math.atan2(t3, t4)



    return X, Y, Z



def fe(actual):

    new = pd.DataFrame()

    actual['total_angular_velocity'] = (actual['angular_velocity_X'] ** 2 + actual['angular_velocity_Y'] ** 2 + actual['angular_velocity_Z'] ** 2) ** 0.5

    actual['total_linear_acceleration'] = (actual['linear_acceleration_X'] ** 2 + actual['linear_acceleration_Y'] ** 2 + actual['linear_acceleration_Z'] ** 2) ** 0.5

    

    actual['acc_vs_vel'] = actual['total_linear_acceleration'] / actual['total_angular_velocity']

    

    x, y, z, w = actual['orientation_X'].tolist(), actual['orientation_Y'].tolist(), actual['orientation_Z'].tolist(), actual['orientation_W'].tolist()

    nx, ny, nz = [], [], []

    for i in range(len(x)):

        xx, yy, zz = quaternion_to_euler(x[i], y[i], z[i], w[i])

        nx.append(xx)

        ny.append(yy)

        nz.append(zz)

    

    actual['euler_x'] = nx

    actual['euler_y'] = ny

    actual['euler_z'] = nz

    

    actual['total_angle'] = (actual['euler_x'] ** 2 + actual['euler_y'] ** 2 + actual['euler_z'] ** 2) ** 5

    actual['angle_vs_acc'] = actual['total_angle'] / actual['total_linear_acceleration']

    actual['angle_vs_vel'] = actual['total_angle'] / actual['total_angular_velocity']

    

    def f1(x):

        return np.mean(np.diff(np.abs(np.diff(x))))

    

    def f2(x):

        return np.mean(np.abs(np.diff(x)))

    

    for col in actual.columns:

        if col in ['row_id', 'series_id', 'measurement_number']:

            continue

        new[col + '_mean'] = actual.groupby(['series_id'])[col].mean()

        new[col + '_min'] = actual.groupby(['series_id'])[col].min()

        new[col + '_max'] = actual.groupby(['series_id'])[col].max()

        new[col + '_std'] = actual.groupby(['series_id'])[col].std()

        new[col + '_max_to_min'] = new[col + '_max'] / new[col + '_min']



    return new
X_train = fe(X_train)

X_test = fe(X_test)
n_fold = 20

folds = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=11)
le = LabelEncoder()

le.fit(y_train['surface'])

y_train['surface'] = le.transform(y_train['surface'])
def eval_acc(preds, dtrain):

    labels = dtrain.get_label()

    return 'acc', accuracy_score(labels, preds.argmax(1)), True



def train_model(X, X_test, y, params=None, folds=folds, model_type='lgb', plot_feature_importance=False, model=None, groups=y_train['group_id']):



    oof = np.zeros((len(X), 9))

    prediction = np.zeros((len(X_test), 9))

    scores = []

    feature_importance = pd.DataFrame()

    for fold_n, (train_index, valid_index) in enumerate(folds.split(X, y, groups)):

        print('Fold', fold_n, 'started at', time.ctime())

        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]

        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

        

        if model_type == 'lgb':

            model = lgb.LGBMClassifier(**params, n_estimators = 10000, n_jobs = -1)

            model.fit(X_train, y_train, 

                    eval_set=[(X_train, y_train), (X_valid, y_valid)], eval_metric='multi_logloss',

                    verbose=5000, early_stopping_rounds=200)

            

            y_pred_valid = model.predict_proba(X_valid)

            y_pred = model.predict_proba(X_test, num_iteration=model.best_iteration_)

            

        if model_type == 'xgb':

            train_data = xgb.DMatrix(data=X_train, label=y_train, feature_names=X.columns)

            valid_data = xgb.DMatrix(data=X_valid, label=y_valid, feature_names=X.columns)



            watchlist = [(train_data, 'train'), (valid_data, 'valid_data')]

            model = xgb.train(dtrain=train_data, num_boost_round=20000, evals=watchlist, early_stopping_rounds=200, verbose_eval=500, params=params)

            y_pred_valid = model.predict(xgb.DMatrix(X_valid, feature_names=X.columns), ntree_limit=model.best_ntree_limit)

            y_pred = model.predict(xgb.DMatrix(X_test, feature_names=X.columns), ntree_limit=model.best_ntree_limit)

        

        if model_type == 'sklearn':

            model = model

            model.fit(X_train, y_train)

            

            y_pred_valid = model.predict_proba(X_valid)

            score = accuracy_score(y_valid, y_pred_valid.argmax(1))

            print(f'Fold {fold_n}. Accuracy: {score:.4f}.')

            print('')

            

            y_pred = model.predict_proba(X_test)

        

        if model_type == 'cat':

            model = CatBoostClassifier(iterations=20000,  eval_metric='MAE', **params)

            model.fit(X_train, y_train, eval_set=(X_valid, y_valid), cat_features=[], use_best_model=True, verbose=False)



            y_pred_valid = model.predict(X_valid)

            y_pred = model.predict(X_test)

        

        oof[valid_index] = y_pred_valid

        scores.append(accuracy_score(y_valid, y_pred_valid.argmax(1)))



        prediction += y_pred    

        

        if model_type == 'lgb':

            # feature importance

            fold_importance = pd.DataFrame()

            fold_importance["feature"] = X.columns

            fold_importance["importance"] = model.feature_importances_

            fold_importance["fold"] = fold_n + 1

            feature_importance = pd.concat([feature_importance, fold_importance], axis=0)



    prediction /= n_fold

    

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

        

            return oof, prediction, feature_importance

        return oof, prediction

    

    else:

        return oof, prediction
params = {'num_leaves': 123,

          'min_data_in_leaf': 12,

          'objective': 'multiclass',

          'max_depth': 20,

          'learning_rate': 0.04680350949723872,

          "boosting": "gbdt",

          "bagging_freq": 5,

          "bagging_fraction": 0.8933018355190274,

          "bagging_seed": 11,

          "verbosity": -1,

          'reg_alpha': 0.9498109326932401,

          'reg_lambda': 0.8058490960546196,

          "num_class": 9,

          'nthread': -1,

          'min_split_gain': 0.009913227240564853,

          'subsample': 0.9027358830703129

         }



oof_lgb, prediction_lgb, feature_importance = train_model(X=X_train, X_test=X_test, y=y_train['surface'], params=params, model_type='lgb', plot_feature_importance=True)
import itertools



def plot_confusion_matrix(truth, pred, classes, normalize=False, title=''):

    cm = confusion_matrix(truth, pred)

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    

    plt.figure(figsize=(10, 10))

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

    plt.title('Confusion matrix', size=15)

    plt.colorbar(fraction=0.046, pad=0.04)

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, format(cm[i, j], fmt),

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    plt.grid(False)

    plt.tight_layout()
plot_confusion_matrix(y_train['surface'], oof_lgb.argmax(1), le.classes_)
model = lgb.LGBMClassifier(**params, n_estimators = 20000, n_jobs = -1)

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train['surface'], test_size=0.2, stratify=y_train['surface'])

model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_valid, y_valid)], verbose=5000, early_stopping_rounds=200)
eli5.show_weights(model, targets=[0, 1], feature_names=list(X_train.columns), top=40, feature_filter=lambda x: x != '<BIAS>')
sample_submission['surface'] = le.inverse_transform(prediction_lgb.argmax(1))

sample_submission.to_csv('lgb_sub.csv', index=False)