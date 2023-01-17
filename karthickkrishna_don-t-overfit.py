# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd 

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')

import pandas_profiling as pp

import xgboost as xgb

# import lightgbm as lgb

from pandas import get_dummies

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn import linear_model

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import RepeatedStratifiedKFold

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn.metrics import classification_report

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import confusion_matrix

# from catboost import CatBoostClassifier,Pool

from sklearn.metrics import accuracy_score

from sklearn.decomposition import PCA

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

train_data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')



# Any results you write to the current directory are saved as output.
train_data.info()
pp.ProfileReport(train_data)
train_data.shape
train_data.isnull().any().any()
train_data['target'].value_counts()
plt.figure(figsize=(15,6))

sns.countplot(train_data['target'])
train_data['target'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',shadow=True)

plt.title('target')
X_train = train_data.drop(['id','target'],axis=1)

y_train = train_data['target']

X_test = test_data.drop(['id'],axis=1)



#Scaling

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
ridge = linear_model.Ridge()

lasso = linear_model.Lasso()

elastic = linear_model.ElasticNet()

lasso_lars = linear_model.LassoLars()

bayesian_ridge = linear_model.BayesianRidge()

logistic = linear_model.LogisticRegression(solver='liblinear')

sgd = linear_model.SGDClassifier()
models = [ridge, lasso, elastic, lasso_lars, bayesian_ridge, logistic, sgd]
def get_cv_scores(model):

    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')

    print('CV Mean: ', np.mean(scores))

    print('STD: ', np.std(scores))

    print('\n')
for model in models:

    print(model)

    get_cv_scores(model)
penalty = ['l1', 'l2']

C = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]

class_weight = [{1:0.5, 0:0.5}, {1:0.4, 0:0.6}, {1:0.6, 0:0.4}, {1:0.7, 0:0.3}]

solver = ['liblinear', 'saga']



param_grid = dict(penalty=penalty,

                  C=C,

                  class_weight=class_weight,

                  solver=solver)



grid = GridSearchCV(estimator=logistic, param_grid=param_grid, scoring='roc_auc', verbose=1, n_jobs=-1)

grid_result = grid.fit(X_train, y_train)



print('Best Score: ', grid_result.best_score_)

print('Best Params: ', grid_result.best_params_)
logistic = linear_model.LogisticRegression(C=1, class_weight={1:0.6, 0:0.4}, penalty='l1', solver='liblinear')

get_cv_scores(logistic)
predictions = logistic.fit(X_train, y_train).predict_proba(X_test)
predictions
submission = pd.read_csv('../input/sample_submission.csv')

submission['target'] = predictions

submission.head()
loss = ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron']

penalty = ['l1', 'l2', 'elasticnet']

alpha = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]

learning_rate = ['constant', 'optimal', 'invscaling', 'adaptive']

class_weight = [{1:0.5, 0:0.5}, {1:0.4, 0:0.6}, {1:0.6, 0:0.4}, {1:0.7, 0:0.3}]

eta0 = [1, 10, 100]



param_distributions = dict(loss=loss,

                           penalty=penalty,

                           alpha=alpha,

                           learning_rate=learning_rate,

                           class_weight=class_weight,

                           eta0=eta0)



random = RandomizedSearchCV(estimator=sgd, param_distributions=param_distributions, scoring='roc_auc', verbose=1, n_jobs=-1, n_iter=1000)

random_result = random.fit(X_train, y_train)



print('Best Score: ', random_result.best_score_)

print('Best Params: ', random_result.best_params_)
sgd = linear_model.SGDClassifier(alpha=0.1,

                                 class_weight={1:0.7, 0:0.3},

                                 eta0=100,

                                 learning_rate='optimal',

                                 loss='log',

                                 penalty='elasticnet')

get_cv_scores(sgd)
predictions = sgd.fit(X_train, y_train).predict_proba(X_test)
predictions