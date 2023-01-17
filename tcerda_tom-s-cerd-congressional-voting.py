# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



sns.set(rc={'figure.figsize': (14, 12)})



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv('/kaggle/input/mse-3-bb-ds-ws19-congressional-voting/CongressionalVotingID.shuf.train.csv', na_values='unknown')

test_df = pd.read_csv('/kaggle/input/mse-3-bb-ds-ws19-congressional-voting/CongressionalVotingID.shuf.test.csv', na_values='unknown')

train_df.head()
train_df = train_df.replace(['republican', 'democrat', 'y', 'n'], [1, 0, 1, 0]).drop(['ID'], axis=1)

test_df = test_df.replace(['republican', 'democrat', 'y', 'n'], [1, 0, 1, 0])
print('Train samples:')

print(str(len(train_df)))



print('\nTrain dataset:')

print(train_df.isna().sum()[train_df.isna().any()])



print('\nTest samples:')

print(str(len(test_df)))



print('\nTest dataset:')

print(test_df.isna().sum()[test_df.isna().any()])
train_df.describe()
corr = train_df.corr()

sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True);
X = train_df.drop(columns=['class'])

y = train_df['class']
import lightgbm as lgb

from sklearn.model_selection import GridSearchCV



estimator = lgb.LGBMClassifier()



param_grid = {

    'num_leaves': [2, 3, 7, 15],

    'max_depth': [2, 4, 8, 16, -1],

    'learning_rate': [0.001, 0.01, 0.02, 0.05, 0.1],

    'n_estimators': [50, 100, 300, 350, 400, 500, 550],

    'objective': ['binary']

}



gbm = GridSearchCV(estimator, param_grid, cv=5)

gbm.fit(X.drop(columns=['water-project-cost-sharing', 'export-administration-act-south-africa']), y)



std = gbm.cv_results_['std_test_score'][gbm.best_index_]



print(f'Best params: {gbm.best_params_}')

print(f'Best score: {gbm.best_score_} (+/- {std})')

# 0.9724
from sklearn.svm import SVC



svc_param_grid = {

    'C': np.arange(0.10, 0.90, 0.01),

    'gamma': ['scale', 'auto'],

    'shrinking': [True, False],

    'kernel': ['linear', 'rbf', 'sigmoid']

}



svc_estimator = SVC()



svc = GridSearchCV(cv=5, estimator=svc_estimator, param_grid=svc_param_grid)

svc.fit(X.fillna(X.mean()).drop(columns=['water-project-cost-sharing', 'export-administration-act-south-africa']), y)



svc_std = svc.cv_results_['std_test_score'][svc.best_index_]



print(f'Best params: {svc.best_params_}')

print(f'Best score: {svc.best_score_} (+/- {svc_std})')

# 0.9633
clf = gbm.best_estimator_

predictions = clf.predict(test_df.drop(columns=['ID', 'water-project-cost-sharing', 'export-administration-act-south-africa']))

ans = pd.DataFrame({'ID': test_df['ID'], 'class': predictions})

ans['class'] = ans['class'].replace([1, 0], ['republican', 'democrat'])

ans.to_csv('gbm_submission.csv', index=False)