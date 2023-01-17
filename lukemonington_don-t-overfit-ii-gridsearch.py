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

from sklearn.feature_selection import GenericUnivariateSelect, SelectPercentile, SelectKBest, f_classif, mutual_info_classif, RFE

import statsmodels.api as sm

import warnings

warnings.filterwarnings('ignore')

from catboost import CatBoostClassifier
train = pd.read_csv('../input/dont-overfit-ii/train.csv')

test = pd.read_csv('../input/dont-overfit-ii/test.csv')

labels = train.columns.drop(['id', 'target'])

test_id=test['id']

test_features = test.drop(['id'],axis=1)

train.head()
test.head()
print("Train Shape: " , train.shape , "\nTest Shape:" , test.shape)
train['target'].value_counts().plot(kind='bar', title='Count (target)')
train[train.columns[2:]].std().plot('hist')

plt.title('Distribution of the Standard Deviations of the Features')
train[train.columns[2:]].mean().plot('hist')

plt.title('Distribution of the Means of the Features')
print('Distributions of the first 28 columns')

plt.figure(figsize=(26,24))

for i, col in enumerate(list(train.columns)[2:30]):

    plt.subplot(7,4,i+1)

    plt.hist(train[col])

    plt.title(col)
X_train = train.drop(['id','target'], axis=1)

y_train = train['target']

X_test = test.drop(['id'], axis=1)
from imblearn.over_sampling import SMOTE

smote = SMOTE(ratio='minority', n_jobs=-1)

X_sm, y_sm = smote.fit_resample(X_train, y_train)

#outputs X_sm and y_sm as ndarrays, need to convert back to df

X_train = pd.DataFrame(X_sm, columns=labels)

y_train = pd.DataFrame(y_sm, columns=['target'])
cols = X_train.columns

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_train = pd.DataFrame(X_train, columns = cols)

X = X_train

y = y_train



X_test = scaler.transform(X_test)

X_test = pd.DataFrame(X_test, columns = cols)
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)
y_t = pd.Series(y_train.iloc[:,0], name="training")

y_t.value_counts().plot('bar')

plt.title('Count of y_train target variable after SMOTE')
from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

from sklearn.model_selection import ShuffleSplit, GridSearchCV

from sklearn.metrics import log_loss
cv_sets = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)

classifiers = [RandomForestClassifier(), SVC(), KNeighborsClassifier()]

params = [{'n_estimators': [3, 10,30]},

         {'kernel':('linear','poly','sigmoid','rbf'), 'C':[0.01,0.05,0.025,0.07,0.09,1.0], 'gamma':['scale'], 'probability':[True]},

         {'n_neighbors': [3,5,7,9]}]
best_estimators = []

for classifier, param in zip(classifiers,params):

    grid = GridSearchCV(classifier,param,cv=cv_sets)

    grid = grid.fit(X_train,y_train)

    best_estimators.append(grid.best_estimator_)
for estimator in best_estimators:

    estimator.fit(X_train, y_train)

    name = estimator.__class__.__name__

    

    print("="*30)

    print(name)

    

    print('****Results****')

    print('**Training set**')

    train_predictions = estimator.predict(X_train)

    acc = accuracy_score(y_train, train_predictions)

    print("Accuracy: {:.4%}".format(acc))

    train_predictions = estimator.predict_proba(X_train)

    ll = log_loss(y_train, train_predictions)

    print("Log Loss: {}".format(ll))

    

    print('**Validation set**')

    train_predictions = estimator.predict(X_val)

    acc = accuracy_score(y_val, train_predictions)

    print("Accuracy: {:.4%}".format(acc))

    train_predictions = estimator.predict_proba(X_val)

    ll = log_loss(y_val, train_predictions)

    print("Log Loss: {}".format(ll))

    

print("="*30)
pred = best_estimators[1].predict(X_test)
submission = pd.DataFrame(pred, index = test_id, columns=['target'])
submission
submission['target']=submission['target'].astype('int64')
submission.to_csv('submission.csv', index=False)