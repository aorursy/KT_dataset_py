# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import xgboost
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_auc_score

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
path = '../input/cat-in-the-dat/'
train = pd.read_csv(path+'train.csv')
test = pd.read_csv(path+'test.csv')
submission = pd.read_csv(path+'sample_submission.csv')
train.head()
train.describe()
target_var = train['target']
train.drop(['id', 'target'], axis=1, inplace=True)
test.drop(['id'], axis=1, inplace=True)
encoder = LabelEncoder()
categorical_features = train.columns.tolist()
for each in categorical_features:
    train[each] = encoder.fit_transform(train[each])
    
test_cat_features = test.columns.tolist()
for col in test_cat_features:
    test[col] = encoder.fit_transform(test[col])
# feature scaling
scaler = StandardScaler()
train = scaler.fit_transform(train)
test = scaler.transform(test)
xgb_clf = xgboost.XGBClassifier(n_estimators=500, 
                            n_jobs=-1, 
                            subsample=0.7,
                            max_depth=7,
                            reg_alpha=0.2,
                            reg_lambda=1, 
                            random_state=1, 
                            learning_rate=0.1,
                            metric = 'auc',
                            tree_method= 'gpu_hist', 
                            objective="binary:logistic",
                            verbose=100,
                            early_stopping_rounds=200)
X = train
y = target_var
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=1)
roc_scorer = make_scorer(roc_auc_score)
xgb_clf.fit(X_train, y_train)
predictions = xgb_clf.predict_proba(X_val)[:,1]
pipeline_rf = Pipeline([
    ('model', RandomForestClassifier(max_depth=11, n_jobs=-1, random_state=1))
])
param_grid_rf = {'model__n_estimators': [75, 100, 150, 200, 250, 300]
                 }

grid_rf = GridSearchCV(estimator=pipeline_rf, param_grid=param_grid_rf,
                       scoring=roc_scorer, n_jobs=-1,
                       cv=3, verbose=1, return_train_score=False)

grid_rf.fit(X_train, y_train)
# xgboost evaluation
score = roc_auc_score(y_val,predictions)
print(score)
# randomforest evaluation
preds = grid_rf.predict_proba(X_val)[:,1]
roc = roc_auc_score(y_val, preds)
print(roc)
submission['target'] = xgb_clf.predict_proba(test)[:,1]
submission.to_csv('submission.csv', index=False)
