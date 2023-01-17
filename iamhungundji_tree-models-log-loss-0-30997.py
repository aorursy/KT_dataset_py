import pandas as pd

import numpy as np
train = pd.read_csv('/kaggle/input/machinehack-financial-risk-prediction/Train.csv')

test = pd.read_csv('/kaggle/input/machinehack-financial-risk-prediction/Test.csv')
train.shape, test.shape
train.head()
combine = train.append(test)

combine.shape
combine['City'].value_counts()
combine['Location_Score'].describe()
combine['External_Audit_Score'].value_counts()
combine['Internal_Audit_Score'].value_counts()
combine['Fin_Score'].value_counts()
combine['Loss_score'].value_counts()
combine['Past_Results'].value_counts()
combine.columns
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

cols = ['External_Audit_Score', 'Fin_Score', 'Internal_Audit_Score',

        'Location_Score', 'Loss_score', 'Past_Results']



combine[cols] = scaler.fit_transform(combine[cols])
combine.head()
X = combine[combine['IsUnderRisk'].isnull()!=True].drop(['IsUnderRisk'], axis=1)

y = combine[combine['IsUnderRisk'].isnull()!=True]['IsUnderRisk'].reset_index(drop=True)



X_test = combine[combine['IsUnderRisk'].isnull()==True].drop(['IsUnderRisk'], axis=1)



X.shape, y.shape, X_test.shape
from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
from lightgbm import LGBMClassifier

model = LGBMClassifier(boosting_type='gbdt',

                       max_depth=5,

                       learning_rate=0.05,

                       n_estimators=5000,

                       min_child_weight=0.01,

                       colsample_bytree=0.5,

                       num_leaves=30,

                       random_state=1994)



model.fit(x_train,y_train,

          eval_set=[(x_train,y_train),(x_val, y_val.values)],

          eval_metric='log_loss',

          early_stopping_rounds=100,

          verbose=200)



pred_y = model.predict_proba(x_val)
from sklearn.metrics import log_loss

log_loss(y_val, pred_y)
pred_test = model.predict_proba(X_test)
import lightgbm

import matplotlib.pyplot as plt

%matplotlib inline

lightgbm.plot_importance(model)
X.shape
err = []

y_pred_tot_lgm = []



from sklearn.model_selection import StratifiedKFold



fold = StratifiedKFold(n_splits=15, shuffle=True, random_state=2020)

i = 1

for train_index, test_index in fold.split(X, y):

    x_train, x_val = X.iloc[train_index], X.iloc[test_index]

    y_train, y_val = y[train_index], y[test_index]

    m = LGBMClassifier(boosting_type='gbdt',

                       max_depth=5,

                       learning_rate=0.05,

                       n_estimators=5000,

                       min_child_weight=0.01,

                       colsample_bytree=0.5,

                       num_leaves=30,

                       random_state=1994)

    m.fit(x_train, y_train,

          eval_set=[(x_train,y_train),(x_val, y_val)],

          early_stopping_rounds=200,

          eval_metric='log_loss',

          verbose=200)

    pred_y = m.predict_proba(x_val)

    print("err_lgm: ",log_loss(y_val,pred_y))

    err.append(log_loss(y_val, pred_y))

    pred_test = m.predict_proba(X_test)

    i = i + 1

    y_pred_tot_lgm.append(pred_test)
np.mean(err,0)
from xgboost import XGBClassifier



errxgb = []

y_pred_tot_xgb = []



from sklearn.model_selection import KFold,StratifiedKFold



fold = StratifiedKFold(n_splits=15, shuffle=True, random_state=2020)

i = 1

for train_index, test_index in fold.split(X,y):

    x_train, x_val = X.iloc[train_index], X.iloc[test_index]

    y_train, y_val = y[train_index], y[test_index]

    m = XGBClassifier(boosting_type='gbdt',

                      max_depth=5,

                      learning_rate=0.07,

                      n_estimators=5000,

                      random_state=1994)

    m.fit(x_train, y_train,

          eval_set=[(x_train,y_train),(x_val, y_val)],

          early_stopping_rounds=200,

          eval_metric='logloss',

          verbose=200)

    pred_y = m.predict_proba(x_val)

    print("err_xgb: ",log_loss(y_val,pred_y))

    errxgb.append(log_loss(y_val, pred_y))

    pred_test = m.predict_proba(X_test)

    i = i + 1

    y_pred_tot_xgb.append(pred_test)
np.mean(errxgb,0)
from catboost import CatBoostClassifier,Pool, cv

errCB = []

y_pred_tot_cb = []

from sklearn.model_selection import KFold,StratifiedKFold



fold = StratifiedKFold(n_splits=15, shuffle=True, random_state=2020)

i = 1

for train_index, test_index in fold.split(X,y):

    x_train, x_val = X.iloc[train_index], X.iloc[test_index]

    y_train, y_val = y[train_index], y[test_index]

    m = CatBoostClassifier(n_estimators=1000,

                           random_state=1994,

                           eval_metric='Logloss',

                           learning_rate=0.5, 

                           max_depth=10)

    m.fit(x_train, y_train,

          eval_set=[(x_train,y_train),(x_val, y_val)],

          early_stopping_rounds=200,

          verbose=200)

    pred_y = m.predict_proba(x_val)

    print("err_cb: ",log_loss(y_val,pred_y))

    errCB.append(log_loss(y_val,pred_y))

    pred_test = m.predict_proba(X_test)

    i = i + 1

    y_pred_tot_cb.append(pred_test)
np.mean(errCB, 0)
(np.mean(errxgb, 0) + np.mean(err, 0) + np.mean(errCB, 0))/3
submission = pd.DataFrame((np.mean(y_pred_tot_lgm, 0)+np.mean(y_pred_tot_xgb, 0)+np.mean(y_pred_tot_cb, 0))/3)

submission.to_excel('Submission.xlsx', index=False)
submission.shape
submission.head()