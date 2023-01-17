import numpy as np

import pandas as pd

import matplotlib.pyplot as plt 

import seaborn as sns; sns.set()
import os

print(os.listdir('../input'))
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

sub = pd.read_csv('../input/sample_submission.csv')
train.head()
test.head()
X_train = train.drop(['id', 'target'], axis=1).values

y_train = train['target'].values



X_test = test.drop(['id'], 1).values
from sklearn.preprocessing import QuantileTransformer

scaler = QuantileTransformer()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
X_train.shape, y_train.shape, X_test.shape
from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC
import warnings

warnings.filterwarnings("ignore")
svm = SVC(C=100, kernel='linear', max_iter=100, gamma='auto', probability=True, random_state=0)

svm.fit(X_train, y_train)
from sklearn.model_selection import cross_val_score



score = cross_val_score(svm, X_train, y_train, cv=20, scoring='roc_auc')

print(score)

print('-' * 60)

print(score.max())
from sklearn.model_selection import GridSearchCV



lr = LogisticRegression(solver='liblinear', max_iter=1000).fit(X_train, y_train)



parameter_grid = {'class_weight' : ['balanced', None],

                  'penalty' : ['l2'],

                  'C' : [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],

                  'solver': ['newton-cg', 'sag', 'lbfgs']

                 }



grid_search = GridSearchCV(lr, param_grid=parameter_grid, cv=20, scoring='roc_auc')

grid_search.fit(X_train, y_train)

print('Best score: {}'.format(grid_search.best_score_))

print('Best parameters: {}'.format(grid_search.best_params_))
svm_pred = svm.predict_proba(X_test)[:, 1]

lr_pred = lr.predict_proba(X_test)[:, 1]



av_pred = (svm_pred + lr_pred) / 2
sub.head()
sub['target'] = svm_pred

sub.to_csv('svm_submission.csv', index=False)
sub['target'] = lr_pred

sub.to_csv('lr_submission.csv', index=False)
sub['target'] = av_pred

sub.to_csv('submission.csv', index=False)