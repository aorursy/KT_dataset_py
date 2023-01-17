# Libraries

import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



pd.set_option('max_columns', None)



from scipy import stats

from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, cross_val_score, GridSearchCV, RepeatedStratifiedKFold

from sklearn.preprocessing import StandardScaler



import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls

import xgboost as xgb

import lightgbm as lgb



from sklearn import model_selection

from sklearn.metrics import accuracy_score, roc_auc_score

from sklearn import metrics

from sklearn import linear_model



import json

import ast



import os

import time

import datetime



import eli5

from eli5.sklearn import PermutationImportance

import shap

from tqdm import tqdm_notebook



from mlxtend.feature_selection import SequentialFeatureSelector as SFS

from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs

from sklearn.neighbors import NearestNeighbors

from sklearn.feature_selection import GenericUnivariateSelect, SelectPercentile, SelectKBest, f_classif, mutual_info_classif, RFE



import statsmodels.api as sm

import warnings

warnings.filterwarnings('ignore')

from catboost import CatBoostClassifier
# Read in data

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

                   
#df = pd.DataFrame({'value': np.random.randint(1, 80, 20)}) 

#df['group'] = pd.cut(df.value,

#                     bins=[0, 5, 31, 51, 80],

#                     labels=["very short", "short", "long", "very long"])
#df['group']
sns.set_style("whitegrid")
#test['var_0bk'] = pd.cut(test.var_0, bins=8,labels=[1,2,3,4,5,6,7,8])

#pd.value_counts(test['var_0bk'])
plt.figure(figsize=(10,6))

ax = sns.boxplot(x=train['var_0'])
plt.figure(figsize=(12,8))

train.var_40.hist(bins=20)
#train['var_3N'] = np.square(train['var_3'])
#plt.figure(figsize=(10,6))

#ax = sns.boxplot(x=train['var_3N'])
#plt.figure(figsize=(12,8))

#train.var_3N.hist(bins=20)
#train['var_5'].quantile(0.50)
#train = train.drop(columns=['var_new_feat'])

#train['var_new_feat'] = np.sqrt(train['var_110'])

#train.var_new_feat.hist(bins=20)
#train[train['var_5'] <  0.924800].count()
#train.var_5.min()

#var = ['var_0','var_1']

#train[var].max()
#(train['var_0'].max()) - (11.5006)
#train[train['ID_code'] == 'train_1']
#train['var_4N'] = np.square(train['var_4'])
idx = features = train.columns.values[2:202]

for df in [test, train]:

    df['sum'] = df[idx].sum(axis=1)  

    df['min'] = df[idx].min(axis=1)

    df['max'] = df[idx].max(axis=1)

    df['mean'] = df[idx].mean(axis=1)

    df['std'] = df[idx].std(axis=1)

    df['skew'] = df[idx].skew(axis=1)

    df['kurt'] = df[idx].kurtosis(axis=1)

    df['med'] = df[idx].median(axis=1)
test[test.columns[201:]].head()
train['target'].value_counts(normalize=True)
X = train.drop(['ID_code', 'target'], axis=1)

y = train['target']

X_test = test.drop(['ID_code'], axis=1)

n_fold = 10

folds = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=42)

repeated_folds = RepeatedStratifiedKFold(n_splits=10, n_repeats=20, random_state=42)
def train_model(X, X_test, y, params, folds, model_type='lgb', plot_feature_importance=False, averaging='usual', model=None):

    oof = np.zeros(len(X))

    prediction = np.zeros(len(X_test))

    scores = []

    feature_importance = pd.DataFrame()

    for fold_n, (train_index, valid_index) in enumerate(folds.split(X, y)):

        print('Fold', fold_n, 'started at', time.ctime())

        X_train, X_valid = X.loc[train_index], X.loc[valid_index]

        y_train, y_valid = y[train_index], y[valid_index]

        

        if model_type == 'lgb':

            train_data = lgb.Dataset(X_train, label=y_train)

            valid_data = lgb.Dataset(X_valid, label=y_valid)

            

            model = lgb.train(params,

                    train_data,

                    num_boost_round=20000,

                    valid_sets = [train_data, valid_data],

                    verbose_eval=1000,

                    early_stopping_rounds = 200)

            

            y_pred_valid = model.predict(X_valid)

            y_pred = model.predict(X_test, num_iteration=model.best_iteration)

            

        if model_type == 'xgb':

            train_data = xgb.DMatrix(data=X_train, label=y_train, feature_names=X_train.columns)

            valid_data = xgb.DMatrix(data=X_valid, label=y_valid, feature_names=X_train.columns)



            watchlist = [(train_data, 'train'), (valid_data, 'valid_data')]

            model = xgb.train(dtrain=train_data, num_boost_round=20000, evals=watchlist, early_stopping_rounds=200, verbose_eval=500, params=params)

            y_pred_valid = model.predict(xgb.DMatrix(X_valid, feature_names=X_train.columns), ntree_limit=model.best_ntree_limit)

            y_pred = model.predict(xgb.DMatrix(X_test, feature_names=X_train.columns), ntree_limit=model.best_ntree_limit)

        

        if model_type == 'sklearn':

            model = model

            model.fit(X_train, y_train)

            y_pred_valid = model.predict_proba(X_valid).reshape(-1,)

            score = roc_auc_score(y_valid, y_pred_valid)

            # print(f'Fold {fold_n}. AUC: {score:.4f}.')

            # print('')

            

            y_pred = model.predict_proba(X_test)[:, 1]

            

        if model_type == 'glm':

            model = sm.GLM(y_train, X_train, family=sm.families.Binomial())

            model_results = model.fit()

            model_results.predict(X_test)

            y_pred_valid = model_results.predict(X_valid).reshape(-1,)

            score = roc_auc_score(y_valid, y_pred_valid)

            

            y_pred = model_results.predict(X_test)

            

        if model_type == 'cat':

            model = CatBoostClassifier(iterations=20000, learning_rate=0.05, loss_function='Logloss',  eval_metric='AUC', **params)

            model.fit(X_train, y_train, eval_set=(X_valid, y_valid), cat_features=[], use_best_model=True, verbose=False)



            y_pred_valid = model.predict_proba(X_valid)[:, 1]

            y_pred = model.predict_proba(X_test)[:, 1]

            

        oof[valid_index] = y_pred_valid.reshape(-1,)

        scores.append(roc_auc_score(y_valid, y_pred_valid))



        if averaging == 'usual':

            prediction += y_pred

        elif averaging == 'rank':

            prediction += pd.Series(y_pred).rank().values  

        

        if model_type == 'lgb':

            # feature importance

            fold_importance = pd.DataFrame()

            fold_importance["feature"] = X.columns

            fold_importance["importance"] = model.feature_importance()

            fold_importance["fold"] = fold_n + 1

            feature_importance = pd.concat([feature_importance, fold_importance], axis=0)



    prediction /= n_fold

    

    print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))

    

    if model_type == 'lgb':

        feature_importance["importance"] /= n_fold

        if plot_feature_importance:

            cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(

                by="importance", ascending=False)[:100].index



            best_features = feature_importance.loc[feature_importance.feature.isin(cols)]



            plt.figure(figsize=(16, 12));

            sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False));

            plt.title('LGB Features (avg over folds)');

        

            return oof, prediction, feature_importance

        return oof, prediction, scores

    

    else:

        return oof, prediction, scores
params = {'num_leaves': 8,

         'min_data_in_leaf': 42,

         'objective': 'binary',

         'max_depth': 16,

         'learning_rate': 0.0123,

         'boosting': 'gbdt',

         'bagging_freq': 5,

         'feature_fraction': 0.8201,

         'bagging_seed': 11,

         'reg_alpha': 1.728910519108444,

         'reg_lambda': 4.9847051755586085,

         'random_state': 42,

         'metric': 'auc',

         'verbosity': -1,

         'subsample': 0.81,

         'min_gain_to_split': 0.01077313523861969,

         'min_child_weight': 19.428902804238373,

         'num_threads': 4}

oof_lgb, prediction_lgb, scores = train_model(X, X_test, y, params=params, folds=folds, model_type='lgb', plot_feature_importance=True)
sub = pd.read_csv('../input/sample_submission.csv')

sub['target'] = prediction_lgb

sub.to_csv('lgb3.csv', index=False)
#predict_lable = train.target
#from sklearn.model_selection import train_test_split

#x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 0)
#from sklearn.linear_model import LogisticRegression

#logreg = LogisticRegression()
#train.info()
#logreg.fit(x_train,y_train)
#y_predict = logreg.predict(x_test)
#from sklearn import metrics

#metrics.accuracy_score(y_test, y_predict)
#sub_predict = logreg.predict(test1)
#var_53

#var_34

#var_174

#var_78

#var_22

#var_33

#var_21

#var_6

#var_165

#var_76

#var_166

#var_81

#var_1

#var_169

#var_190

#var_99

#var_146

#var_109

#var_92

#var_13

#var_198

#var_133

#var_139

#var_184

#var_108

#var_94

#var_110

#var_40

#var_12

#var_80

#var_2

#var_26

#var_173

#var_121

#var_177

#var_122

#var_9

#var_170

#var_44

#var_191

#var_67

#var_118

#var_0

#var_154

#var_164

#var_127

#var_91

#var_179

#var_18

#var_56