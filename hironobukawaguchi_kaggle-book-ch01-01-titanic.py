# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('../input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')
train_x = train.drop(['Survived'], axis=1)

train_y = train['Survived']
test_x = test.copy()
from sklearn.preprocessing import LabelEncoder
train_x = train_x.drop(['PassengerId'], axis=1)

test_x = test_x.drop(['PassengerId'], axis=1)
train_x = train_x.drop(['Name', 'Ticket', 'Cabin'], axis=1)

test_x = test_x.drop(['Name', 'Ticket', 'Cabin'], axis=1)
for c in ['Sex', 'Embarked']:

    le = LabelEncoder()

    le.fit(train_x[c].fillna('NA'))

    train_x[c] = le.transform(train_x[c].fillna('NA'))

    test_x[c] = le.transform(test_x[c].fillna('NA'))
from xgboost import XGBClassifier
model = XGBClassifier(n_estimators=20, random_state=71)

model.fit(train_x, train_y)
pred = model.predict_proba(test_x)[:, 1]
pred_label = np.where(pred > 0.5, 1, 0)
submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': pred_label})

submission.to_csv('submission_first.csv', index=False)
from sklearn.metrics import log_loss, accuracy_score

from sklearn.model_selection import KFold
scores_accuracy = []

scores_logloss = []
kf = KFold(n_splits=4, shuffle=True, random_state=71)

for tr_idx, va_idx in kf.split(train_x):

    tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]

    tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]

    model = XGBClassifier(n_estimators=20, random_state=71)

    model.fit(tr_x, tr_y)

    va_pred = model.predict_proba(va_x)[:, 1]

    logloss = log_loss(va_y, va_pred)

    accuracy = accuracy_score(va_y, va_pred > 0.5)

    scores_logloss.append(logloss)

    scores_accuracy.append(accuracy)
logloss = np.mean(scores_logloss)

accuracy = np.mean(scores_accuracy)

print(f'logloss: {logloss:.4f}, accuracy: {accuracy:.4f}')
import itertools
param_space = {

    'max_depth': [3, 5, 7],

    'min_child_weight': [1.0, 2.0, 4.0]

}
param_combinations = itertools.product(param_space['max_depth'], param_space['min_child_weight'])
params = []

scores = []
for max_depth, min_child_weight in param_combinations:

    score_folds = []

    kf = KFold(n_splits=4, shuffle=True, random_state=123456)

    for tr_idx, va_idx in kf.split(train_x):

        tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]

        tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]

        model = XGBClassifier(n_estimators=20, random_state=71, max_depth=max_depth, min_child_weight=min_child_weight)

        model.fit(tr_x, tr_y)

        va_pred = model.predict_proba(va_x)[:, 1]

        logloss = log_loss(va_y, va_pred)

        score_folds.append(logloss)

    score_mean = np.mean(score_folds)

    params.append((max_depth, min_child_weight))

    scores.append(score_mean)
best_idx = np.argsort(scores)[0]

best_param = params[best_idx]

print(f'max_depth: {best_param[0]}, min_child_weight: {best_param[1]}')
from sklearn.preprocessing import OneHotEncoder
train_x2 = train.drop(['Survived'], axis=1)

test_x2 = test.copy()
train_x2 = train_x2.drop(['PassengerId'], axis=1)

test_x2 = test_x2.drop(['PassengerId'], axis=1)
train_x2 = train_x2.drop(['Name', 'Ticket', 'Cabin'], axis=1)

test_x2 = test_x2.drop(['Name', 'Ticket', 'Cabin'], axis=1)
cat_cols = ['Sex', 'Embarked', 'Pclass']

ohe = OneHotEncoder(categories='auto', sparse=False)

ohe.fit(train_x2[cat_cols].fillna('NA'))
ohe_columns = []

for i, c in enumerate(cat_cols):

    ohe_columns += [f'{c}_{v}' for v in ohe.categories_[i]]
ohe_columns
ohe_train_x2 = pd.DataFrame(ohe.transform(train_x2[cat_cols].fillna('NA')), columns=ohe_columns)

ohe_test_x2 = pd.DataFrame(ohe.transform(test_x2[cat_cols].fillna('NA')), columns=ohe_columns)
train_x2 = train_x2.drop(cat_cols, axis=1)

test_x2 = test_x2.drop(cat_cols, axis=1)
train_x2 = pd.concat([train_x2, ohe_train_x2], axis=1)

test_x2 = pd.concat([test_x2, ohe_test_x2], axis=1)
num_cols = ['Age', 'SibSp', 'Parch', 'Fare']

for col in num_cols:

    train_x2[col].fillna(train_x2[col].mean(), inplace=True)

    test_x2[col].fillna(train_x2[col].mean(), inplace=True)
train_x2['Fare'] = np.log1p(train_x2['Fare'])

test_x2['Fare'] = np.log1p(test_x2['Fare'])
from sklearn.linear_model import LogisticRegression
model_xgb = XGBClassifier(n_estimators=20, random_state=71)

model_xgb.fit(train_x, train_y)

pred_xgb = model_xgb.predict_proba(test_x)[:, 1]
model_lr = LogisticRegression(solver='lbfgs', max_iter=300)

model_lr.fit(train_x2, train_y)

pred_lr = model_lr.predict_proba(test_x2)[:, 1]
pred = pred_xgb * 0.8 + pred_lr * 0.2

pred_label = np.where(pred > 0.5, 1, 0)
submission = pd.DataFrame({'passengerId': test['PassengerId'], 'Survived': pred_label})

submission.to_csv('submission_ensemble.csv', index=False)