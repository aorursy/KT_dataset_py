import numpy as np

import pandas as pd

pd.set_option('max_columns', None)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from matplotlib.colors import ListedColormap

import datetime

import lightgbm as lgb

from scipy import stats

from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, cross_val_score, GridSearchCV, RepeatedStratifiedKFold

from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier

from sklearn.feature_selection import SelectFromModel

from sklearn.metrics import accuracy_score

import os

from scipy.stats import zscore

from sklearn import metrics

from sklearn.model_selection import KFold

from keras.layers.core import Dense, Activation

import tensorflow as tf

from tensorflow import keras

import keras

from sklearn.datasets import make_moons

from keras.callbacks import EarlyStopping

from keras.callbacks import ModelCheckpoint

from keras.models import load_model

from keras.models import Sequential

from keras.layers import Dense

from keras import models

from keras import layers

from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import cross_val_score

from sklearn.datasets import make_classification

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

import gc

import itertools

from mlxtend.feature_selection import SequentialFeatureSelector as SFS

from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs

from sklearn.neighbors import NearestNeighbors

from sklearn.feature_selection import GenericUnivariateSelect, SelectPercentile, SelectKBest, f_classif, mutual_info_classif, RFE

import statsmodels.api as sm

import warnings

warnings.filterwarnings('ignore')

from catboost import CatBoostClassifier
IS_LOCAL = False

if (IS_LOCAL):

    location = "../input/dont-overfit/"

else:

    location = "../input/"

os.listdir(location)
%%time

train = pd.read_csv(os.path.join(location, 'train.csv'))

test = pd.read_csv(os.path.join(location, 'test.csv'))

sample_submission = pd.read_csv(os.path.join(location, 'sample_submission.csv'))
print("train: {}\ntest: {}".format(train.shape, test.shape))
def show_head(data):

    return(data.head())
show_head(train)
show_head(test)
def missing_data(data):

    total = data.isnull().sum()

    percent = (total/data.isnull().count()*100)

    miss_column = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

    types = []

    for col in data.columns:

        dtype = str(data[col].dtype)

        types.append(dtype)

    miss_column['Types'] = types

    return(np.transpose(miss_column)) 
missing_data(train)
missing_data(test)
def describe_data(data):

    return(data.describe())
describe_data(train)
describe_data(test)
label = train['target']

train = train.drop(['id', 'target'], axis=1)

test = test.drop(['id'], axis=1)
label_count = label.value_counts().reset_index().rename(columns = {'index' : 'Labels'})

label_count
show_head(train)
show_head(label)
sc = StandardScaler()

xtrain = sc.fit_transform(train)

xtest = sc.transform(test)



pca = PCA().fit(xtrain)

plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.xlabel('number of components')

plt.ylabel('cumulative explained variance');
print("xtrain: {}\nxtest: {}\nlabel: {}".format(xtrain.shape, xtest.shape, label.shape))
missing_data(train)
from sklearn.metrics import classification_report as c_report

from sklearn.metrics import confusion_matrix as c_matrix

folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=4422)

forcast_rfc2 = np.zeros(len(xtest))

validation_pred_rfc2 = np.zeros(len(xtrain))

scores_rfc2 = []

valid_rfc2 = np.zeros(len(label))

for fold_, (xtrn_indx, xvad_indx) in enumerate(folds.split(xtrain, label)):

    print('Fold {}, started at {}'.format(fold_, time.ctime()))

    x_train, x_valid = xtrain[xtrn_indx], xtrain[xvad_indx]

    y_train, y_valid = label[xtrn_indx], label[xvad_indx]

    

    clf_rfc2 = RandomForestClassifier(n_estimators=1000, n_jobs=-1, random_state=0)

    clf_rfc2.fit(x_train, y_train)

    pred_valid = clf_rfc2.predict(x_valid).reshape(-1,)

    score = accuracy_score(y_valid, pred_valid)

    Pred_Real = clf_rfc2.predict_proba(xtest)[:, 1]

    

    validation_pred_rfc2[xvad_indx] += (pred_valid).reshape(-1,)

    scores_rfc2.append(accuracy_score(y_valid, pred_valid))

    forcast_rfc2 += Pred_Real

    valid_rfc2[xvad_indx] += (y_valid)



print(c_report(valid_rfc2[xvad_indx], validation_pred_rfc2[xvad_indx]))

print(c_matrix(valid_rfc2[xvad_indx], validation_pred_rfc2[xvad_indx]))

print('accuracy is: {}, std: {}.'.format(np.mean(scores_rfc2), np.std(scores_rfc2)))
from sklearn.tree import DecisionTreeClassifier

forcast_dtc2 = np.zeros(len(xtest))

validation_pred_dtc2 = np.zeros(len(xtrain))

scores_dtc2 = []

valid_dtc2 = np.zeros(len(label))

for fold_, (xtrn_indx, xvad_indx) in enumerate(folds.split(xtrain, label)):

    print('Fold {}, started at {}'.format(fold_, time.ctime()))

    x_train, x_valid = xtrain[xtrn_indx], xtrain[xvad_indx]

    y_train, y_valid = label[xtrn_indx], label[xvad_indx]

    

    clf_dtc2 = DecisionTreeClassifier()

    clf_dtc2.fit(x_train, y_train)

    pred_valid = clf_dtc2.predict(x_valid).reshape(-1,)

    score = accuracy_score(y_valid, pred_valid)

    Pred_Real = clf_dtc2.predict_proba(xtest)[:, 1]

    

    validation_pred_dtc2[xvad_indx] += (pred_valid).reshape(-1,)

    scores_dtc2.append(accuracy_score(y_valid, pred_valid))

    forcast_dtc2 += Pred_Real

    valid_dtc2[xvad_indx] += (y_valid)



print(c_report(valid_dtc2[xvad_indx], validation_pred_dtc2[xvad_indx]))

print(c_matrix(valid_dtc2[xvad_indx], validation_pred_dtc2[xvad_indx]))

print('accuracy is: {}, std: {}.'.format(np.mean(scores_dtc2), np.std(scores_dtc2)))
from sklearn import svm

forcast_svm2 = np.zeros(len(xtest))

validation_pred_svm2 = np.zeros(len(xtrain))

scores_svm2 = []

valid_svm2 = np.zeros(len(label))

for fold_, (xtrn_indx, xvad_indx) in enumerate(folds.split(xtrain, label)):

    print('Fold {}, started at {}'.format(fold_, time.ctime()))

    x_train, x_valid = xtrain[xtrn_indx], xtrain[xvad_indx]

    y_train, y_valid = label[xtrn_indx], label[xvad_indx]

    

    clf_svm2 = svm.SVC(kernel='linear', gamma=1)

    clf_svm2.fit(x_train, y_train)

    pred_valid = clf_svm2.predict(x_valid).reshape(-1,)

    score = accuracy_score(y_valid, pred_valid)

    Pred_Real = clf_svm2.predict(xtest)

    

    validation_pred_svm2[xvad_indx] += (pred_valid).reshape(-1,)

    scores_svm2.append(accuracy_score(y_valid, pred_valid))

    forcast_svm2 += Pred_Real

    valid_svm2[xvad_indx] += (y_valid)



print(c_report(valid_svm2[xvad_indx], validation_pred_svm2[xvad_indx]))

print(c_matrix(valid_svm2[xvad_indx], validation_pred_svm2[xvad_indx]))

print('accuracy is: {}, std: {}.'.format(np.mean(scores_svm2), np.std(scores_svm2)))
forcast_lr2 = np.zeros(len(xtest))

validation_pred_lr2 = np.zeros(len(xtrain))

scores_lr2 = []

valid_lr2 = np.zeros(len(label))

for fold_, (xtrn_indx, xvad_indx) in enumerate(folds.split(xtrain, label)):

    print('Fold {}, started at {}'.format(fold_, time.ctime()))

    x_train, x_valid = xtrain[xtrn_indx], xtrain[xvad_indx]

    y_train, y_valid = label[xtrn_indx], label[xvad_indx]

    

    clf_lr2 = linear_model.LogisticRegression(class_weight='balanced', penalty='l1', C=0.1, solver='liblinear')

    clf_lr2.fit(x_train, y_train)

    pred_valid = clf_lr2.predict(x_valid).reshape(-1,)

    score = accuracy_score(y_valid, pred_valid)

    Pred_Real = clf_lr2.predict_proba(xtest)[:, 1]

    

    validation_pred_lr2[xvad_indx] += (pred_valid).reshape(-1,)

    scores_lr2.append(accuracy_score(y_valid, pred_valid))

    forcast_lr2 += Pred_Real

    valid_lr2[xvad_indx] += (y_valid)



print(c_report(valid_lr2[xvad_indx], validation_pred_lr2[xvad_indx]))

print(c_matrix(valid_lr2[xvad_indx], validation_pred_lr2[xvad_indx]))

print('accuracy is: {}, std: {}.'.format(np.mean(scores_lr2), np.std(scores_lr2)))
perm = PermutationImportance(clf_lr2, random_state=1).fit(xtrain, label)

eli5.show_weights(perm, top=50)
(clf_lr2.coef_ != 0).sum()
top_features = [i[1:] for i in eli5.formatters.as_dataframe.explain_weights_df(clf_lr2).feature if 'BIAS' not in i]

xtrain_lr_elif = train[top_features]

xtest_lr_elif = test[top_features]

scaler = StandardScaler()

xtrain_lr_elif = scaler.fit_transform(xtrain_lr_elif)

xtest_lr_elif = scaler.transform(xtest_lr_elif)



forcast_lr4 = np.zeros(len(xtest_lr_elif))

validation_pred_lr4 = np.zeros(len(xtrain_lr_elif))

scores_lr4 = []

valid_lr4 = np.zeros(len(label))

for fold_, (xtrn_indx, xvad_indx) in enumerate(folds.split(xtrain_lr_elif, label)):

    print('Fold {}, started at {}'.format(fold_, time.ctime()))

    x_train, x_valid = xtrain_lr_elif[xtrn_indx], xtrain_lr_elif[xvad_indx]

    y_train, y_valid = label[xtrn_indx], label[xvad_indx]

    

    clf_lr4 = linear_model.LogisticRegression(class_weight='balanced', penalty='l1', C=0.1, solver='liblinear')

    clf_lr4.fit(x_train, y_train)

    pred_valid = clf_lr4.predict(x_valid).reshape(-1,)

    score = accuracy_score(y_valid, pred_valid)

    Pred_Real = clf_lr4.predict_proba(xtest_lr_elif)[:, 1]

    

    validation_pred_lr4[xvad_indx] += (pred_valid).reshape(-1,)

    scores_lr4.append(accuracy_score(y_valid, pred_valid))

    forcast_lr4 += Pred_Real

    valid_lr4[xvad_indx] += (y_valid)



print(c_report(valid_lr4[xvad_indx], validation_pred_lr4[xvad_indx]))

print(c_matrix(valid_lr4[xvad_indx], validation_pred_lr4[xvad_indx]))

print('accuracy is: {}, std: {}.'.format(np.mean(scores_lr4), np.std(scores_lr4)))
perm = PermutationImportance(clf_lr2, random_state=1).fit(xtrain, label)

eli5.show_weights(perm, top=50)
top_features = [i[1:] for i in eli5.formatters.as_dataframe.explain_weights_df(perm).feature if 'BIAS' not in i]

xtrain_lr_perm = train[top_features]

xtest_lr_perm = test[top_features]

scaler = StandardScaler()

xtrain_lr_perm = scaler.fit_transform(xtrain_lr_perm)

xtest_lr_perm = scaler.transform(xtest_lr_perm)



forcast_lr3 = np.zeros(len(xtest_lr_perm))

validation_pred_lr3 = np.zeros(len(xtrain_lr_perm))

scores_lr3 = []

valid_lr3 = np.zeros(len(label))

for fold_, (xtrn_indx, xvad_indx) in enumerate(folds.split(xtrain_lr_perm, label)):

    print('Fold {}, started at {}'.format(fold_, time.ctime()))

    x_train, x_valid = xtrain_lr_perm[xtrn_indx], xtrain_lr_perm[xvad_indx]

    y_train, y_valid = label[xtrn_indx], label[xvad_indx]

    

    clf_lr3 = linear_model.LogisticRegression(class_weight='balanced', penalty='l1', C=0.1, solver='liblinear')

    clf_lr3.fit(x_train, y_train)

    pred_valid = clf_lr3.predict(x_valid).reshape(-1,)

    score = accuracy_score(y_valid, pred_valid)

    Pred_Real = clf_lr3.predict_proba(xtest_lr_perm)[:, 1]

    

    validation_pred_lr3[xvad_indx] += (pred_valid).reshape(-1,)

    scores_lr3.append(accuracy_score(y_valid, pred_valid))

    forcast_lr3 += Pred_Real

    valid_lr3[xvad_indx] += (y_valid)



print(c_report(valid_lr3[xvad_indx], validation_pred_lr3[xvad_indx]))

print(c_matrix(valid_lr3[xvad_indx], validation_pred_lr3[xvad_indx]))

print('accuracy is: {}, std: {}.'.format(np.mean(scores_lr3), np.std(scores_lr3)))
from sklearn.ensemble import BaggingClassifier

forcast_bc = np.zeros(len(xtest))

validation_pred_bc = np.zeros(len(xtrain))

scores_bc = []

valid_bc = np.zeros(len(label))

for fold_, (xtrn_indx, xvad_indx) in enumerate(folds.split(xtrain, label)):

    print('Fold {}, started at {}'.format(fold_, time.ctime()))

    x_train, x_valid = xtrain[xtrn_indx], xtrain[xvad_indx]

    y_train, y_valid = label[xtrn_indx], label[xvad_indx]

    

    clf_bc = BaggingClassifier()

    clf_bc.fit(x_train, y_train)

    pred_valid = clf_bc.predict(x_valid).reshape(-1,)

    score = accuracy_score(y_valid, pred_valid)

    Pred_Real = clf_bc.predict_proba(xtest)[:, 1]

    

    validation_pred_bc[xvad_indx] += (pred_valid).reshape(-1,)

    scores_bc.append(accuracy_score(y_valid, pred_valid))

    forcast_bc += Pred_Real

    valid_bc[xvad_indx] += (y_valid)



print(c_report(valid_bc[xvad_indx], validation_pred_bc[xvad_indx]))

print(c_matrix(valid_bc[xvad_indx], validation_pred_bc[xvad_indx]))

print('accuracy is: {}, std: {}.'.format(np.mean(scores_bc), np.std(scores_bc)))
from sklearn.ensemble import AdaBoostClassifier

forcast_adac = np.zeros(len(xtest))

validation_pred_adac = np.zeros(len(xtrain))

scores_adac = []

valid_adac = np.zeros(len(label))

for fold_, (xtrn_indx, xvad_indx) in enumerate(folds.split(xtrain, label)):

    print('Fold {}, started at {}'.format(fold_, time.ctime()))

    x_train, x_valid = xtrain[xtrn_indx], xtrain[xvad_indx]

    y_train, y_valid = label[xtrn_indx], label[xvad_indx]

    

    clf_adac = AdaBoostClassifier()

    clf_adac.fit(x_train, y_train)

    pred_valid = clf_adac.predict(x_valid).reshape(-1,)

    score = accuracy_score(y_valid, pred_valid)

    Pred_Real = clf_adac.predict_proba(xtest)[:, 1]

    

    validation_pred_adac[xvad_indx] += (pred_valid).reshape(-1,)

    scores_adac.append(accuracy_score(y_valid, pred_valid))

    forcast_adac += Pred_Real

    valid_adac[xvad_indx] += (y_valid)



print(c_report(valid_adac[xvad_indx], validation_pred_adac[xvad_indx]))

print(c_matrix(valid_adac[xvad_indx], validation_pred_adac[xvad_indx]))

print('accuracy is: {}, std: {}.'.format(np.mean(scores_adac), np.std(scores_adac)))
from sklearn.ensemble import GradientBoostingClassifier

forcast_gbc = np.zeros(len(xtest))

validation_pred_gbc = np.zeros(len(xtrain))

scores_gbc = []

valid_gbc = np.zeros(len(label))

for fold_, (xtrn_indx, xvad_indx) in enumerate(folds.split(xtrain, label)):

    print('Fold {}, started at {}'.format(fold_, time.ctime()))

    x_train, x_valid = xtrain[xtrn_indx], xtrain[xvad_indx]

    y_train, y_valid = label[xtrn_indx], label[xvad_indx]

    

    clf_gbc = GradientBoostingClassifier()

    clf_gbc.fit(x_train, y_train)

    pred_valid = clf_gbc.predict(x_valid).reshape(-1,)

    score = accuracy_score(y_valid, pred_valid)

    Pred_Real = clf_gbc.predict_proba(xtest)[:, 1]

    

    validation_pred_gbc[xvad_indx] += (pred_valid).reshape(-1,)

    scores_gbc.append(accuracy_score(y_valid, pred_valid))

    forcast_gbc += Pred_Real

    valid_gbc[xvad_indx] += (y_valid)



print(c_report(valid_gbc[xvad_indx], validation_pred_gbc[xvad_indx]))

print(c_matrix(valid_gbc[xvad_indx], validation_pred_gbc[xvad_indx]))

print('accuracy is: {}, std: {}.'.format(np.mean(scores_gbc), np.std(scores_gbc)))
cat_params = {'learning_rate': 0.02,

              'depth': 5,

              'l2_leaf_reg': 10,

              'bootstrap_type': 'Bernoulli',

              'od_type': 'Iter',

              'od_wait': 50,

              'random_seed': 11,

              'allow_writing_files': False}

forcast_catc = np.zeros(len(xtest))

validation_pred_catc = np.zeros(len(xtrain))

scores_catc = []

valid_catc = np.zeros(len(label))

for fold_, (xtrn_indx, xvad_indx) in enumerate(folds.split(xtrain, label)):

    x_train, x_valid = xtrain[xtrn_indx], xtrain[xvad_indx]

    y_train, y_valid = label[xtrn_indx], label[xvad_indx]

    

    clf_catc = CatBoostClassifier(iterations=400, **cat_params)

    clf_catc.fit(x_train, y_train)

    pred_valid = clf_catc.predict(x_valid).reshape(-1,)

    score = accuracy_score(y_valid, pred_valid)

    Pred_Real = clf_catc.predict_proba(xtest)[:, 1]

    

    validation_pred_catc[xvad_indx] += (pred_valid).reshape(-1,)

    scores_catc.append(accuracy_score(y_valid, pred_valid))

    forcast_catc += Pred_Real

    valid_catc[xvad_indx] += (y_valid)



print(c_report(valid_catc[xvad_indx], validation_pred_catc[xvad_indx]))

print(c_matrix(valid_catc[xvad_indx], validation_pred_catc[xvad_indx]))

print('accuracy is: {}, std: {}.'.format(np.mean(scores_catc), np.std(scores_catc)))
sample_submission['target'] = forcast_lr4

sample_submission.to_csv('Forcasting.csv', index=False)

sample_submission.head(20)