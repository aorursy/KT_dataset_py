import numpy as np

import os

import pandas as pd

import sys

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB

from sklearn.svm import LinearSVC,SVC

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, ExtraTreesClassifier, AdaBoostClassifier, BaggingClassifier

from sklearn.linear_model import SGDClassifier, LogisticRegression

from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV

from sklearn.preprocessing import StandardScaler

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import cross_val_score

from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, roc_auc_score, recall_score, precision_score,make_scorer

from sklearn.decomposition import PCA

import seaborn as sns

sns.set(color_codes=True)

from scipy import stats

from scipy.stats import norm, skew #for some statistics

import xgboost as xgb

import lightgbm as lgb

from catboost import CatBoostClassifier

np.random.seed(25)

import os
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.head()
target = train['target']
train.isnull().sum()
sns.countplot(train['target'])
print(any(train.duplicated())) 
correlations = train.corr().abs().unstack().sort_values(kind="quicksort").reset_index()

correlations = correlations[correlations['level_0'] != correlations['level_1']]

correlations.tail(10)
feature_names = [x for x in train.columns if x not in ['ID_code','target']]
# model = CatBoostClassifier(iterations=5000,eval_metric='AUC',random_seed=42,verbose=False,task_type='GPU')

# print(cross_val_score(model, train_pca, target, cv=5, scoring=make_scorer(roc_auc_score)))
model = CatBoostClassifier(iterations=5000,learning_rate=0.01,eval_metric='AUC',random_seed=42,verbose=False,task_type='GPU')

## model training and prediction

model.fit(train[feature_names],target)

pred1 = model.predict_proba(test[feature_names])
model = lgb.LGBMClassifier(n_estimators=5000,n_jobs = -1,learning_rate=0.01)

## model training and prediction

model.fit(train[feature_names],target)

pred2 = model.predict_proba(test[feature_names])
model = xgb.XGBClassifier(n_estimators=5000,n_jobs = -1,learning_rate=0.01)

## model training and prediction

model.fit(train[feature_names],target)

pred3 = model.predict_proba(test[feature_names])
model = RandomForestClassifier(n_estimators=1000,n_jobs = -1)

## model training and prediction

model.fit(train[feature_names],target)

pred4 = model.predict_proba(test[feature_names])
pred = []

for i in range(len(pred1)):

    pred.append((pred1[i][1] + pred2[i][1] + pred3[i][1] + pred4[i][1])/4)
## make submission

sub = pd.DataFrame()

sub['ID_code'] = test['ID_code']

sub['target'] = pred

sub.to_csv('result.csv', index=False)
sub.head()