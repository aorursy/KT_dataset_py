# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import lightgbm as lgb

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import GridSearchCV , train_test_split , cross_val_score

from sklearn.linear_model import LogisticRegression



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# import Dataset to play with it

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

sample_submission = pd.read_csv('../input/sample_submission.csv')
sample_submission.head()
#defining new variable for target

target = train['target']
#dropping the id column

train = train.drop(['id','target'], axis=1)

test = test.drop('id', axis=1)
# from sklearn.model_selection import train_test_split

# X_train, X_val, y_train, y_val = train_test_split(train, target, test_size = 0.1, random_state = 0)
from sklearn.preprocessing import  StandardScaler

sc = StandardScaler()
#Scaling the data

X_train = sc.fit_transform(train)

# X_val = sc.fit_transform(X_val)

X_test = sc.fit_transform(test)
# based on following kernel https://www.kaggle.com/dromosys/sctp-working-lgb

#parameters for the LightGBM model

params = {'num_leaves': 9,

         'min_data_in_leaf': 42,

         'objective': 'binary',

         'max_depth': 16,

         'learning_rate': 0.0123,

         'boosting': 'gbdt',

         'bagging_freq': 5,

         'bagging_fraction': 0.8,

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
#LGB Model

train_data = lgb.Dataset(X_train, label=target)



model = lgb.train(params,

        train_data,

        num_boost_round=2000,

        valid_sets = [train_data],

        verbose_eval=500,

        early_stopping_rounds = 200)



y_pred_lgb = model.predict(X_test, num_iteration=model.best_iteration)
#logreg

clf = LogisticRegression(class_weight='balanced', solver='liblinear', penalty ='l1', C= 0.1, max_iter=10000)

clf.fit(X_train, target)

y_pred_logreg = clf.predict_proba(X_test)
#lgb

sample_submission['target'] = y_pred_lgb

sample_submission.to_csv('submission_lgb.csv', index=False)



#logreg

sample_submission['target'] = y_pred_logreg[:,1]

sample_submission.to_csv('submission_logreg.csv', index=False)
