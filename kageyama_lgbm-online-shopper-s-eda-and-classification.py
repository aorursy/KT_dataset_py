import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns



import matplotlib.pyplot as plt

%matplotlib inline

from tqdm import tqdm_notebook

from sklearn.preprocessing import StandardScaler

from sklearn.svm import NuSVC, SVC

from sklearn.metrics import accuracy_score, classification_report,confusion_matrix

pd.options.display.precision = 15



import lightgbm as lgb

import xgboost as xgb

import time

import datetime



import json

import ast

import eli5

import shap

from eli5.sklearn import PermutationImportance

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold, train_test_split

from sklearn.linear_model import Ridge, RidgeCV

import gc

from catboost import CatBoostClassifier

import seaborn as sns

import warnings

warnings.filterwarnings("ignore")



import altair as alt

from IPython.display import HTML

from sklearn.linear_model import LinearRegression



from scipy.signal import hilbert

from scipy.signal import hann

from scipy.signal import convolve

from scipy import stats

from sklearn.kernel_ridge import KernelRidge



from bayes_opt import BayesianOptimization



import os

IS_LOCAL = False

if(IS_LOCAL):

    PATH="../input/Santander/"

else:

    PATH="../input/"

os.listdir(PATH)
%%time

df = pd.read_csv(PATH+"online_shoppers_intention.csv")
df.shape
df.head()
def missing_data(data):

    total = data.isnull().sum()

    percent = (data.isnull().sum()/data.isnull().count()*100)

    tt = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

    types = []

    for col in data.columns:

        dtype = str(data[col].dtype)

        types.append(dtype)

    tt['Types'] = types

    return(np.transpose(tt))
%%time

missing_data(df)
df = df.dropna()
%%time

df.describe()
df.dtypes
# categorical feature

df['SpecialDay'] = df['SpecialDay'].astype('object')

df['OperatingSystems'] = df['OperatingSystems'].astype('object')

df['Browser'] = df['Browser'].astype('object')

df['Region'] = df['Region'].astype('object')

df['TrafficType'] = df['TrafficType'].astype('object')
sns.countplot(df['Revenue'], palette='Set3')
print("There are {}% target values with 1".format(100 * df["Revenue"].value_counts()[1]/df.shape[0]))
def plot_feature_distribution(df1, df2, label1, label2, features):

    i = 0

    sns.set_style('whitegrid')

    plt.figure()

    fig, ax = plt.subplots(3,3,figsize=(18,22))



    for feature in features:

        i += 1

        plt.subplot(3,3,i)

        sns.distplot(df1[feature], hist=False,label=label1)

        sns.distplot(df2[feature], hist=False,label=label2)

        plt.xlabel(feature, fontsize=9)

        locs, labels = plt.xticks()

        plt.tick_params(axis='x', which='major', labelsize=6, pad=-6)

        plt.tick_params(axis='y', which='major', labelsize=6)

    plt.show();
t0 = df.loc[df['Revenue'] == True]

t1 = df.loc[df['Revenue'] == False]

features = df.columns.values[0:9]

plot_feature_distribution(t0, t1, 'True', 'False', features)
sns.countplot(df['SpecialDay'], palette='Set3')
sns.countplot(df['OperatingSystems'], palette='Set3')
sns.countplot(df['Browser'], palette='Set3')
sns.countplot(df['Region'], palette='Set3')
sns.countplot(df['TrafficType'], palette='Set3')
sns.countplot(df['Weekend'], palette='Set3')
df = pd.get_dummies(df,columns=['SpecialDay', 'Month', 'OperatingSystems','Browser','Region','TrafficType','VisitorType','Weekend'])
df.columns
target = df['Revenue']

df = df.drop(columns=['Revenue'])
n_fold = 20

folds = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=11)
X_train,X_test,y_train,y_test = train_test_split(df,target,random_state=0)
X_train.shape
X_test.shape
def lgbm_evaluate(**params):

    warnings.simplefilter('ignore')

    

    params['num_leaves'] = int(params['num_leaves'])

    params['max_depth'] = int(params['max_depth'])

        

    clf = lgb.LGBMClassifier(**params, n_estimators=20000, nthread=-1)



    test_pred_proba = np.zeros((X_train.shape[0], 2))

    

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(X_train, y_train)):

        X_train_bo, X_valid = X_train.iloc[train_idx], X_train.iloc[valid_idx]

        y_train_bo, y_valid = y_train.iloc[train_idx], y_train.iloc[valid_idx]

        

        model = lgb.LGBMClassifier(**params, n_estimators = 10000, n_jobs = -1)

        model.fit(X_train_bo, y_train_bo, 

                eval_set=[(X_train_bo, y_train_bo), (X_valid, y_valid)], eval_metric='binary_logloss',

                verbose=False, early_stopping_rounds=200)



        y_pred_valid = model.predict_proba(X_valid)



        test_pred_proba[valid_idx] = y_pred_valid



    return accuracy_score(y_valid, y_pred_valid.argmax(1))
params = {'colsample_bytree': (0.6, 1),

     'learning_rate': (.001, .08), 

      'num_leaves': (8, 124), 

      'subsample': (0.6, 1), 

      'max_depth': (3, 25), 

      'reg_alpha': (.05, 15.0), 

      'reg_lambda': (.05, 15.0), 

      'min_split_gain': (.001, .03),

      'min_child_weight': (12, 80)}



bo = BayesianOptimization(lgbm_evaluate, params)

bo.maximize(init_points=5, n_iter=20)
def eval_acc(preds, dtrain):

    labels = dtrain.get_label()

    return 'acc', accuracy_score(labels, preds.argmax(1)), True



def train_model(X, X_test, y, params=None, folds=folds, model_type='lgb', plot_feature_importance=False, model=None):



    oof = np.zeros((len(X), 2))

    prediction = np.zeros((len(X_test), 2))

    scores = []

    feature_importance = pd.DataFrame()

    for fold_n, (train_index, valid_index) in enumerate(folds.split(X, y)):

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
params = {'num_leaves': int(bo.max['params']['num_leaves']),

          'min_data_in_leaf': int(bo.max['params']['min_child_weight']),

          'min_split_gain': bo.max['params']['min_split_gain'],

          'objective': 'binary',

          'max_depth': int(bo.max['params']['max_depth']),

          'learning_rate': bo.max['params']['learning_rate'],

          "boosting": "gbdt",

          "bagging_freq": 5,

          "bagging_fraction": bo.max['params']['subsample'],

          "bagging_seed": 11,

          "verbosity": -1,

          'reg_alpha': bo.max['params']['reg_alpha'],

          'reg_lambda': bo.max['params']['reg_lambda'],

          "num_class": 1,

          'nthread': -1

         }
oof_lgb, prediction_lgb, feature_importance = train_model(X=X_train, X_test=X_test, y=y_train, params=params, model_type='lgb', plot_feature_importance=True)
print("Test score: ",accuracy_score(y_test,prediction_lgb.argmax(1)))
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
plot_confusion_matrix(y_test, prediction_lgb.argmax(1), ['False','True'])
print(classification_report(y_test, prediction_lgb.argmax(1)))