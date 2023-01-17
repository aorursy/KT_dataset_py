import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns 

%matplotlib inline
train = pd.read_csv('/kaggle/input/novartis-data/Train.csv')

test = pd.read_csv('/kaggle/input/novartis-data/Test.csv')

test1 = pd.read_csv('/kaggle/input/novartis-data/Test.csv')
train.head()
train.shape, test.shape
train.X_12.fillna(2, inplace=True)

test.X_12.fillna(2, inplace=True)
train['DATE'] = pd.to_datetime(train['DATE'], format='%d-%b-%y')

test['DATE'] = pd.to_datetime(test['DATE'], format='%d-%b-%y')

test1['DATE'] = pd.to_datetime(test1['DATE'], format='%d-%b-%y')
for i in (train, test):

    i['YEAR'] = i.DATE.dt.year

    i['MONTH'] = i.DATE.dt.month

    i['DAY'] = i.DATE.dt.day

    i['WEEKDAY'] = i.DATE.dt.dayofweek
YEAR = pd.crosstab(train['YEAR'],train['MULTIPLE_OFFENSE'])

YEAR.div(YEAR.sum(1).astype(float), axis=0).plot(kind="bar", figsize=(15,6))
MONTH = pd.crosstab(train['MONTH'],train['MULTIPLE_OFFENSE'])

MONTH.div(MONTH.sum(1).astype(float), axis=0).plot(kind="bar", figsize=(15,6))
DAY = pd.crosstab(train['DAY'],train['MULTIPLE_OFFENSE'])

DAY.div(DAY.sum(1).astype(float), axis=0).plot(kind="bar", figsize=(15,6))
WEEKDAY = pd.crosstab(train['WEEKDAY'],train['MULTIPLE_OFFENSE'])

WEEKDAY.div(WEEKDAY.sum(1).astype(float), axis=0).plot(kind="bar", figsize=(15,6))
X_1 = pd.crosstab(train['X_1'],train['MULTIPLE_OFFENSE'])

X_1.div(X_1.sum(1).astype(float), axis=0).plot(kind="bar", figsize=(15,6))
X_2 = pd.crosstab(train['X_2'],train['MULTIPLE_OFFENSE'])

X_2.div(X_2.sum(1).astype(float), axis=0).plot(kind="bar", figsize=(15,6))
X_3 = pd.crosstab(train['X_3'],train['MULTIPLE_OFFENSE'])

X_3.div(X_3.sum(1).astype(float), axis=0).plot(kind="bar", figsize=(15,6))
X_4 = pd.crosstab(train['X_4'],train['MULTIPLE_OFFENSE'])

X_4.div(X_4.sum(1).astype(float), axis=0).plot(kind="bar", figsize=(15,6))
X_5 = pd.crosstab(train['X_5'],train['MULTIPLE_OFFENSE'])

X_5.div(X_5.sum(1).astype(float), axis=0).plot(kind="bar", figsize=(15,6))
X_6 = pd.crosstab(train['X_6'],train['MULTIPLE_OFFENSE'])

X_6.div(X_6.sum(1).astype(float), axis=0).plot(kind="bar", figsize=(15,6))
X_7 = pd.crosstab(train['X_7'],train['MULTIPLE_OFFENSE'])

X_7.div(X_7.sum(1).astype(float), axis=0).plot(kind="bar", figsize=(15,6))
X_8 = pd.crosstab(train['X_8'],train['MULTIPLE_OFFENSE'])

X_8.div(X_8.sum(1).astype(float), axis=0).plot(kind="bar", figsize=(15,6))
X_9 = pd.crosstab(train['X_9'],train['MULTIPLE_OFFENSE'])

X_9.div(X_9.sum(1).astype(float), axis=0).plot(kind="bar", figsize=(15,6))
X_10 = pd.crosstab(train['X_10'],train['MULTIPLE_OFFENSE'])

X_10.div(X_10.sum(1).astype(float), axis=0).plot(kind="bar", figsize=(15,6))
X_11 = pd.crosstab(train['X_11'],train['MULTIPLE_OFFENSE'])

X_11.div(X_11.sum(1).astype(float), axis=0).plot(kind="bar", figsize=(25,8))
X_12 = pd.crosstab(train['X_12'],train['MULTIPLE_OFFENSE'])

X_12.div(X_12.sum(1).astype(float), axis=0).plot(kind="bar", figsize=(15,6))
train = train.sort_values(['DATE']).reset_index(drop=True)

test = test.sort_values(['DATE']).reset_index(drop=True)

train.head()
sorted_test_dates = test['DATE']

sorted_test_ids = test['INCIDENT_ID']

train = train.drop(['INCIDENT_ID', 'DATE'], axis=1)

test = test.drop(['INCIDENT_ID', 'DATE'], axis=1)

train.head()
test.head()
from xgboost import XGBClassifier

from lightgbm import LGBMClassifier

from sklearn.model_selection import StratifiedKFold, KFold

from sklearn.metrics import confusion_matrix, recall_score
err = []

y_pred_tot_lgm = []



fold = KFold(n_splits=5)

i = 1



X = train.drop(['MULTIPLE_OFFENSE'], axis=1)

y = train['MULTIPLE_OFFENSE']

X_test = test



for train_index, test_index in fold.split(X, y):

    x_train, x_val = X.iloc[train_index], X.iloc[test_index]

    y_train, y_val = y[train_index], y[test_index]

    m = LGBMClassifier(boosting_type='gbdt',

                       max_depth=5,

                       learning_rate=0.01,

                       n_estimators=5000,

                       min_child_weight=0.01,

                       colsample_bytree=0.5,

                       random_state=1994)

    m.fit(x_train, y_train,

          eval_set=[(x_train,y_train),(x_val, y_val)],

          early_stopping_rounds=200,

          verbose=0)

    pred_y = m.predict(x_val)

    print(i, " err_lgm: ", round(recall_score(y_val, pred_y), 3))

    err.append(recall_score(y_val, pred_y))

    pred_test = m.predict(X_test)

    pred_test_prob = m.predict_proba(X_test)[:,-1]

    i = i + 1

    y_pred_tot_lgm.append(pred_test_prob)

    

plgbm_test = np.mean(y_pred_tot_lgm, 0)
errxgb = []

y_pred_tot_xgb = []



fold = KFold(n_splits=5)

i = 1

for train_index, test_index in fold.split(X, y):

    x_train, x_val = X.iloc[train_index], X.iloc[test_index]

    y_train, y_val = y[train_index], y[test_index]

    m = XGBClassifier(max_depth=5,

                      learning_rate=0.07,

                      n_estimators=5000,

                      random_state=1994)

    m.fit(x_train, y_train,

          eval_set=[(x_train,y_train),(x_val, y_val)],

          early_stopping_rounds=200,

          verbose=0)

    pred_y = m.predict(x_val)

    print(i, " err_xgb: ", round(recall_score(y_val, pred_y), 3))

    errxgb.append(recall_score(y_val, pred_y))

    pred_test = m.predict(X_test)

    pred_test_prob = m.predict_proba(X_test)[:,-1]

    i = i + 1

    y_pred_tot_xgb.append(pred_test_prob)



pxgb_test = np.mean(y_pred_tot_xgb, 0)
(np.mean(err, 0) + np.mean(errxgb)) / 2
pred_test = (pxgb_test+plgbm_test)/2
pred_test = np.array([round(i) for i in pred_test])

submission = pd.DataFrame({'DATE':sorted_test_dates,

                           'INCIDENT_ID': sorted_test_ids,

                           'MULTIPLE_OFFENSE':pred_test})

submission.head()
submission = pd.merge(test1[['DATE','INCIDENT_ID']], 

                      submission, 

                      on=['DATE','INCIDENT_ID'], how='outer')[['INCIDENT_ID', 

                                                  'MULTIPLE_OFFENSE']]

submission.head()
submission['MULTIPLE_OFFENSE'].value_counts()
test.shape, submission.shape
submission.to_csv('submission.csv', index=False)