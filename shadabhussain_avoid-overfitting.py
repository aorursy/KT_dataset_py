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

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls

import xgboost as xgb

import lightgbm as lgb

from sklearn import model_selection

from sklearn.metrics import accuracy_score, roc_auc_score

import json

import ast

import time

from sklearn import linear_model

import eli5

from eli5.sklearn import PermutationImportance

import shap



from mlxtend.feature_selection import SequentialFeatureSelector as SFS

from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs

from sklearn.neighbors import NearestNeighbors
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

train.shape
train.head()
train[train.columns[2:]].std().plot('hist');

plt.title('Distribution of stds of all columns');
train[train.columns[2:]].mean().plot('hist');

plt.title('Distribution of means of all columns');
# we have no missing values

train.isnull().any().any()
print('Distributions of first 28 columns')

plt.figure(figsize=(26, 24))

for i, col in enumerate(list(train.columns)[2:30]):

    plt.subplot(7, 4, i + 1)

    plt.hist(train[col])

    plt.title(col)
train['target'].value_counts()
corrs = train.corr().abs().unstack().sort_values(kind="quicksort").reset_index()

corrs = corrs[corrs['level_0'] != corrs['level_1']]

corrs.tail(10)
X_train = train.drop(['id', 'target'], axis=1)

y_train = train['target']

X_test = test.drop(['id'], axis=1)

n_fold = 10

folds = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=42)

repeated_folds = RepeatedStratifiedKFold(n_splits=10, n_repeats=20)



scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
def train_model(X, X_test, y, params, folds=folds, model_type='lgb', plot_feature_importance=False, averaging='usual', model=None):

    oof = np.zeros(len(X))

    prediction = np.zeros(len(X_test))

    scores = []

    feature_importance = pd.DataFrame()

    for fold_n, (train_index, valid_index) in enumerate(folds.split(X, y)):

        # print('Fold', fold_n, 'started at', time.ctime())

        X_train, X_valid = X[train_index], X[valid_index]

        y_train, y_valid = y[train_index], y[valid_index]

        

        if model_type == 'lgb':

            train_data = lgb.Dataset(X_train, label=y_train)

            valid_data = lgb.Dataset(X_valid, label=y_valid)

            

            model = lgb.train(params,

                    train_data,

                    num_boost_round=2000,

                    valid_sets = [train_data, valid_data],

                    verbose_eval=500,

                    early_stopping_rounds = 200)

            

            y_pred_valid = model.predict(X_valid)

            y_pred = model.predict(X_test, num_iteration=model.best_iteration_)

            

        if model_type == 'xgb':

            train_data = xgb.DMatrix(data=X_train, label=y_train, feature_names=X_tr.columns)

            valid_data = xgb.DMatrix(data=X_valid, label=y_valid, feature_names=X_tr.columns)



            watchlist = [(train_data, 'train'), (valid_data, 'valid_data')]

            model = xgb.train(dtrain=train_data, num_boost_round=20000, evals=watchlist, early_stopping_rounds=200, verbose_eval=500, params=params)

            y_pred_valid = model.predict(xgb.DMatrix(X_valid, feature_names=X_tr.columns), ntree_limit=model.best_ntree_limit)

            y_pred = model.predict(xgb.DMatrix(X_test, feature_names=X_tr.columns), ntree_limit=model.best_ntree_limit)

        

        if model_type == 'sklearn':

            model = model

            model.fit(X_train, y_train)

            y_pred_valid = model.predict(X_valid).reshape(-1,)

            score = roc_auc_score(y_valid, y_pred_valid)

            # print(f'Fold {fold_n}. AUC: {score:.4f}.')

            # print('')

            

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

        return oof, prediction, scores

    

    else:

        return oof, prediction, scores
# A lot of people are using logreg currently, let's try

model = linear_model.LogisticRegression(class_weight='balanced', penalty='l1', C=0.1, solver='liblinear')

oof_lr, prediction_lr, scores = train_model(X_train, X_test, y_train, params=None, model_type='sklearn', model=model)
model = linear_model.LogisticRegression(class_weight='balanced', penalty='l1', C=0.1, solver='liblinear')

oof_lr, prediction_lr_repeated, scores = train_model(X_train, X_test, y_train, params=None, model_type='sklearn', model=model, folds=repeated_folds)
eli5.show_weights(model, top=40)
top_features = [i[1:] for i in eli5.formatters.as_dataframe.explain_weights_df(model).feature if 'BIAS' not in i]

X_train = train[top_features]

X_test = test[top_features]

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
model = linear_model.LogisticRegression(class_weight='balanced', penalty='l1', C=0.1, solver='liblinear')

oof_lr, prediction_lr, _ = train_model(X_train, X_test, y_train, params=None, model_type='sklearn', model=model)
perm = PermutationImportance(model, random_state=1).fit(X_train, y_train)

eli5.show_weights(perm, top=50)
top_features = [i[1:] for i in eli5.formatters.as_dataframe.explain_weights_df(perm).feature if 'BIAS' not in i]

X_train = train[top_features]

X_test = test[top_features]

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
model = linear_model.LogisticRegression(class_weight='balanced', penalty='l1', C=0.1, solver='liblinear')

oof_lr1, prediction_lr1, _ = train_model(X_train, X_test, y_train, params=None, model_type='sklearn', model=model)
X_train = train.drop(['id', 'target'], axis=1)

y_train = train['target']

X_test = test.drop(['id'], axis=1)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)



model = linear_model.LogisticRegression(class_weight='balanced', penalty='l1', C=0.1, solver='liblinear')

oof_lr, prediction_lr, _ = train_model(X_train, X_test, y_train, params=None, model_type='sklearn', model=model)
explainer = shap.LinearExplainer(model, X_train)

shap_values = explainer.shap_values(X_train)



shap.summary_plot(shap_values, X_train)
sfs1 = SFS(model, 

           k_features=(10, 15), 

           forward=True, 

           floating=False, 

           verbose=0,

           scoring='roc_auc',

           cv=folds,

          n_jobs=-1)



sfs1 = sfs1.fit(X_train, y_train)
fig1 = plot_sfs(sfs1.get_metric_dict(), kind='std_dev')



plt.ylim([0.8, 1])

plt.title('Sequential Forward Selection (w. StdDev)')

plt.grid()

plt.show()
top_features = list(sfs1.k_feature_names_)

X_train = train[top_features]

X_test = test[top_features]

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)



model = linear_model.LogisticRegression(class_weight='balanced', penalty='l1', C=0.1, solver='liblinear')

oof_lr, prediction_lr, _ = train_model(X_train, X_test, y_train, params=None, model_type='sklearn', model=model)
X_train = train.drop(['id', 'target'], axis=1)

y_train = train['target']

X_test = test.drop(['id'], axis=1)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
lr = linear_model.LogisticRegression(solver='liblinear', max_iter=1000)



parameter_grid = {'class_weight' : ['balanced', None],

                  'penalty' : ['l2'],

                  'C' : [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],

                  'solver': ['newton-cg', 'sag', 'lbfgs']

                 }



grid_search = GridSearchCV(lr, param_grid=parameter_grid, cv=folds, scoring='roc_auc')

grid_search.fit(X_train, y_train)

print('Best score: {}'.format(grid_search.best_score_))

print('Best parameters: {}'.format(grid_search.best_params_))
lr = linear_model.LogisticRegression(solver='liblinear', max_iter=1000)



parameter_grid = {'class_weight' : ['balanced', None],

                  'penalty' : ['l2', 'l1'],

                  'C' : [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],

                 }



grid_search = GridSearchCV(lr, param_grid=parameter_grid, cv=folds, scoring='roc_auc')

grid_search.fit(X_train, y_train)

print('Best score: {}'.format(grid_search.best_score_))

print('Best parameters: {}'.format(grid_search.best_params_))
model = linear_model.LogisticRegression(class_weight='balanced', penalty='l1', C=0.1, solver='liblinear')

oof_lr, prediction_lr, scores = train_model(X_train, X_test, y_train, params=None, model_type='sklearn', model=model)
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()

oof_gnb, prediction_gnb, scores_gnb = train_model(X_train, X_test, y_train, params=None, model_type='sklearn', model=gnb)
from sklearn.ensemble import AdaBoostClassifier

abc = AdaBoostClassifier()



parameter_grid = {'n_estimators': [5, 10, 20, 50, 100],

                  'learning_rate': [0.001, 0.01, 0.1, 1.0, 10.0]

                 }



grid_search = GridSearchCV(abc, param_grid=parameter_grid, cv=folds, scoring='roc_auc')

grid_search.fit(X_train, y_train)

print('Best score: {}'.format(grid_search.best_score_))

print('Best parameters: {}'.format(grid_search.best_params_))
abc = AdaBoostClassifier(**grid_search.best_params_)

oof_abc, prediction_abc, scores_abc = train_model(X_train, X_test, y_train, params=None, model_type='sklearn', model=abc)
from sklearn.ensemble import ExtraTreesClassifier

etc = ExtraTreesClassifier()



parameter_grid = {'n_estimators': [10, 50, 100, 1000],

                  'max_depth': [None, 3, 5, 15]

                 }



grid_search = GridSearchCV(etc, param_grid=parameter_grid, cv=folds, scoring='roc_auc', n_jobs=-1)

grid_search.fit(X_train, y_train)

print('Best score: {}'.format(grid_search.best_score_))

print('Best parameters: {}'.format(grid_search.best_params_))

etc = ExtraTreesClassifier(**grid_search.best_params_)

oof_etc, prediction_etc, scores_etc = train_model(X_train, X_test, y_train, params=None, model_type='sklearn', model=etc)
from sklearn.gaussian_process import GaussianProcessClassifier

gpc = GaussianProcessClassifier()

oof_gpc, prediction_gpc, scores_gpc = train_model(X_train, X_test, y_train, params=None, model_type='sklearn', model=gpc)
from sklearn.svm import SVC

svc = SVC(probability=True, gamma='scale')



parameter_grid = {'C': [0.001, 0.01, 0.1, 1.0, 10.0],

                  'kernel': ['linear', 'poly', 'rbf'],

                 }



grid_search = GridSearchCV(svc, param_grid=parameter_grid, cv=folds, scoring='roc_auc', n_jobs=-1)

grid_search.fit(X_train, y_train)

print('Best score: {}'.format(grid_search.best_score_))

print('Best parameters: {}'.format(grid_search.best_params_))

svc = SVC(probability=True, gamma='scale', **grid_search.best_params_)

oof_svc, prediction_svc, scores_svc = train_model(X_train, X_test, y_train, params=None, model_type='sklearn', model=svc)
from sklearn.neighbors import KNeighborsClassifier

knc = KNeighborsClassifier()



parameter_grid = {'n_neighbors': [2, 3, 5, 10, 20],

                  'weights': ['uniform', 'distance'],

                  'leaf_size': [5, 10, 30]

                 }



grid_search = GridSearchCV(knc, param_grid=parameter_grid, cv=folds, scoring='roc_auc', n_jobs=-1)

grid_search.fit(X_train, y_train)

print('Best score: {}'.format(grid_search.best_score_))

print('Best parameters: {}'.format(grid_search.best_params_))

knc = KNeighborsClassifier(**grid_search.best_params_)

oof_knc, prediction_knc, scores_knc = train_model(X_train, X_test, y_train, params=None, model_type='sklearn', model=knc)
from sklearn.naive_bayes import BernoulliNB

bnb = BernoulliNB()



parameter_grid = {'alpha': [0.0001, 1, 2, 10]

                 }



grid_search = GridSearchCV(bnb, param_grid=parameter_grid, cv=folds, scoring='roc_auc', n_jobs=-1)

grid_search.fit(X_train, y_train)

print('Best score: {}'.format(grid_search.best_score_))

print('Best parameters: {}'.format(grid_search.best_params_))

bnb = BernoulliNB(**grid_search.best_params_)

oof_bnb, prediction_bnb, scores_bnb = train_model(X_train, X_test, y_train, params=None, model_type='sklearn', model=bnb)
sgd = linear_model.SGDClassifier(eta0=1, max_iter=1000, tol=0.0001)



parameter_grid = {'loss': ['log', 'modified_huber'],

                  'penalty': ['l1', 'l2', 'elasticnet'],

                  'alpha': [0.001, 0.01],

                  'l1_ratio': [0, 0.15, 0.5, 1.0],

                  'learning_rate': ['optimal', 'invscaling', 'adaptive']

                 }



grid_search = GridSearchCV(sgd, param_grid=parameter_grid, cv=folds, scoring='roc_auc', n_jobs=-1)

grid_search.fit(X_train, y_train)

print('Best score: {}'.format(grid_search.best_score_))

print('Best parameters: {}'.format(grid_search.best_params_))

sgd = linear_model.SGDClassifier(eta0=1, tol=0.0001, **grid_search.best_params_)

oof_sgd, prediction_sgd, scores_sgd = train_model(X_train, X_test, y_train, params=None, model_type='sklearn', model=sgd)
plt.figure(figsize=(12, 8));

scores_df = pd.DataFrame({'LogisticRegression': scores})

scores_df['GaussianNB'] = scores_gnb

scores_df['AdaBoostClassifier'] = scores_abc

scores_df['ExtraTreesClassifier'] = scores_etc

scores_df['GaussianProcessClassifier'] = scores_gpc

scores_df['SVC'] = scores_svc

scores_df['KNeighborsClassifier'] = scores_knc

scores_df['BernoulliNB'] = scores_bnb

scores_df['SGDClassifier'] = scores_sgd

sns.boxplot(data=scores_df);

plt.xticks(rotation=45);
X_train = train.drop(['id', 'target'], axis=1)

y_train = train['target']

X_test = test.drop(['id'], axis=1)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)



model = linear_model.LogisticRegression(class_weight='balanced', penalty='l1', C=0.1, solver='liblinear')

oof_lr, prediction_lr, _ = train_model(X_train, X_test, y_train, params=None, model_type='sklearn', model=model)
submission = pd.read_csv('../input/sample_submission.csv')

submission['target'] = (prediction_lr + prediction_svc) / 2

submission.to_csv('submission.csv', index=False)



submission.head()
plt.hist(prediction_lr, label='logreg');

plt.hist(prediction_svc, label='svc');

plt.hist((prediction_lr + prediction_svc) / 2, label='blend');

plt.title('Distribution of out of fold predictions');

plt.legend();
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(2)



X_train = train.drop(['id', 'target'], axis=1)

X_test = test.drop(['id'], axis=1)



X_train_poly = poly.fit_transform(X_train)

X_test_poly = poly.transform(X_test)
cor = pd.DataFrame(X_train_poly).corrwith(y_train)
sc = []

for i in range(10, 510, 10):

    top_corr_cols = list(cor.abs().sort_values().tail(i).reset_index()['index'].values)

    X_train_poly1 = X_train_poly[:, top_corr_cols]

    X_test_poly1 = X_test_poly[:, top_corr_cols]

    oof_lr_poly, prediction_lr_poly, scores = train_model(X_train_poly1, X_test_poly1, y_train, params=None, model_type='sklearn', model=model)

    sc.append(scores)
plt.figure(figsize=(12, 8));

plt.plot([np.mean(i) for i in sc]);

plt.xticks(range(50), range(10, 510, 10), rotation=45);

plt.title('Top poly features vs CV');
top_corr_cols = list(cor.abs().sort_values().tail(200).reset_index()['index'].values)

X_train_poly1 = X_train_poly[:, top_corr_cols]

X_test_poly1 = X_test_poly[:, top_corr_cols]
model = linear_model.LogisticRegression(class_weight='balanced', penalty='l1', C=0.1, solver='liblinear')

oof_lr_poly, prediction_lr_poly, scores = train_model(X_train_poly1, X_test_poly1, y_train, params=None, model_type='sklearn', model=model)
submission = pd.read_csv('../input/sample_submission.csv')

submission['target'] = prediction_lr_poly

submission.to_csv('submission_poly.csv', index=False)



submission.head()
X_train = train.drop(['id', 'target'], axis=1)

X_test = test.drop(['id'], axis=1)

X_train['300'] = X_train.std(1)

X_test['300'] = X_test.std(1)

scaler = StandardScaler()

X_train[X_train.columns[:-1]] = scaler.fit_transform(X_train[X_train.columns[:-1]])

X_test[X_train.columns[:-1]] = scaler.transform(X_test[X_train.columns[:-1]])

model = linear_model.LogisticRegression(class_weight='balanced', penalty='l1', C=0.1, solver='liblinear')

oof_lr_1, prediction_lr_1, scores = train_model(X_train.values, X_test.values, y_train, params=None, model_type='sklearn', model=model)
X_train = train.drop(['id', 'target'], axis=1)

X_test = test.drop(['id'], axis=1)

X_train['300'] = X_train.std(1)

X_test['300'] = X_test.std(1)

scaler = StandardScaler()

X_train[X_train.columns[:-1]] = scaler.fit_transform(X_train[X_train.columns[:-1]])

X_test[X_train.columns[:-1]] = scaler.transform(X_test[X_train.columns[:-1]])

model = linear_model.LogisticRegression(class_weight='balanced', penalty='l1', C=0.1, solver='liblinear')

oof_lr_1, prediction_lr_1_repeated, scores = train_model(X_train.values, X_test.values, y_train, params=None, model_type='sklearn', model=model, folds=repeated_folds)

submission = pd.read_csv('../input/sample_submission.csv')

submission['target'] = prediction_lr_1_repeated

submission.to_csv('repeated_nn_features.csv', index=False)



submission.head()
X_train = train.drop(['id', 'target'], axis=1)

X_test = test.drop(['id'], axis=1)

main_cols = X_train.columns.tolist()
neigh = NearestNeighbors(3, n_jobs=-1)

neigh.fit(X_train)



dists, _ = neigh.kneighbors(X_train, n_neighbors=3)

mean_dist = dists.mean(axis=1)

max_dist = dists.max(axis=1)

min_dist = dists.min(axis=1)



X_train['300'] = X_train.std(1)

X_train = np.hstack((X_train, mean_dist.reshape(-1, 1), max_dist.reshape(-1, 1), min_dist.reshape(-1, 1)))



test_dists, _ = neigh.kneighbors(X_test, n_neighbors=3)



test_mean_dist = test_dists.mean(axis=1)

test_max_dist = test_dists.max(axis=1)

test_min_dist = test_dists.min(axis=1)



X_test['300'] = X_test.std(1)

X_test = np.hstack((X_test, test_mean_dist.reshape(-1, 1), test_max_dist.reshape(-1, 1), test_min_dist.reshape(-1, 1)))
model = linear_model.LogisticRegression(class_weight='balanced', penalty='l1', C=0.1, solver='liblinear')

oof_lr_2, prediction_lr_2, scores = train_model(X_train, X_test, y_train, params=None, model_type='sklearn', model=model)
submission = pd.read_csv('../input/sample_submission.csv')

submission['target'] = (prediction_lr_1 + prediction_lr_2) / 2

submission.to_csv('blend.csv', index=False)



submission.head()