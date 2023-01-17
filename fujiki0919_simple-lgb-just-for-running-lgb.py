import numpy as np

import pandas as pd



import os

print(os.listdir("../input"))
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.head()
test.head()
train.isnull().any()
train['Age'].fillna(train['Age'].mean(), inplace=True)

train['Cabin'].fillna('Missed', inplace=True)

train['Embarked'].fillna('Missed', inplace=True)
test.isnull().any()
test['Age'].fillna(test['Age'].mean(), inplace=True)

test['Fare'].fillna(test['Fare'].mean(), inplace=True)

test['Cabin'].fillna('Missed', inplace=True)
test.isnull().any()
train.duplicated().any()
test.duplicated().any()
train.head()
train.drop(['Name', 'Ticket'], axis=1, inplace=True)
train.head()
test.drop(['Name', 'Ticket'], axis=1, inplace=True)
test.head()
df_sex = pd.get_dummies(train['Sex'], prefix='Sex')

df_cabin = pd.get_dummies(train['Cabin'], prefix='Cabin')

df_embarked = pd.get_dummies(train['Embarked'], prefix='Embarked')
train = pd.concat([train, df_sex, df_cabin, df_embarked], axis=1)

train = train.drop(['Sex', 'Cabin', 'Embarked'], axis=1)
train.head()
df_sex = pd.get_dummies(test['Sex'], prefix='Sex')

df_cabin = pd.get_dummies(test['Cabin'], prefix='Cabin')

df_embarked = pd.get_dummies(test['Embarked'], prefix='Embarked')
test = pd.concat([test, df_sex, df_cabin, df_embarked], axis=1)

test = test.drop(['Sex', 'Cabin', 'Embarked'], axis=1)
test.head()
y = train['Survived']

train = train.drop('Survived', axis=1)

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(train, y)
import lightgbm as lgb

lgb_train = lgb.Dataset(X_train, y_train)

lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)



y_preds = np.zeros((418, 2))

for i in range(3):

    params = {

            'task': 'train',

            'boosting_type': 'gbdt',

            'objective': 'multiclass',

            'metric': {'multi_logloss'},

            'num_class': 2,

            'learning_rate': 0.001,

            'num_leaves': 23,

            'min_data_in_leaf': 1,

            'num_iteration': 100,

            'verbose': 0,

            'random_state': i}

    

    gbm = lgb.train(params,

                lgb_train,

                num_boost_round=100,

                valid_sets=lgb_val,

                early_stopping_rounds=100,

                )

    

    y_pred = gbm.predict(test, num_iteration=gbm.best_iteration)

    y_preds += y_pred    
y_preds
y_preds = np.argmax(y_preds, axis=1)
sub = pd.read_csv('../input/gender_submission.csv')

sub['Survived'] = y_preds

sub.to_csv('submission.csv', index=None)