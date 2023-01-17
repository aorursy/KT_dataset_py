import numpy as np

import pandas as pd



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv('/kaggle/input/av-janata-hack-payment-default-prediction/train_20D8GL3.csv')

test = pd.read_csv('/kaggle/input/av-janata-hack-payment-default-prediction/test_O6kKpvt.csv')

train.shape, test.shape
combine = train.append(test)

combine.shape
combine.isnull().sum()
combine.dtypes
combine['LIMIT_BAL'].describe()
combine['LIMIT_BAL'] = np.log(combine['LIMIT_BAL'])

combine['LIMIT_BAL'].describe()
combine['SEX'].value_counts()
combine['SEX'] = combine['SEX'].replace(1, "Male")

combine['SEX'] = combine['SEX'].replace(2, "Female")

combine['SEX'].value_counts()
combine['EDUCATION'].value_counts()
combine['EDUCATION'] = combine['EDUCATION'].replace(1, "Graduate_School")

combine['EDUCATION'] = combine['EDUCATION'].replace(2, "University")

combine['EDUCATION'] = combine['EDUCATION'].replace(3, "High_School")

combine['EDUCATION'] = combine['EDUCATION'].replace(4, "Unknown")

combine['EDUCATION'] = combine['EDUCATION'].replace(5, "Unknown")

combine['EDUCATION'] = combine['EDUCATION'].replace(6, "Unknown")

combine['EDUCATION'] = combine['EDUCATION'].replace(0, "Unknown")

combine['EDUCATION'].value_counts()
combine['MARRIAGE'].value_counts()
combine['MARRIAGE'] = combine['MARRIAGE'].replace(1, "Married")

combine['MARRIAGE'] = combine['MARRIAGE'].replace(2, "Single")

combine['MARRIAGE'] = combine['MARRIAGE'].replace(3, "Divorced")

combine['MARRIAGE'] = combine['MARRIAGE'].replace(0, "Unknown")

combine['MARRIAGE'].value_counts()
combine['AGE'].describe()
bins= [20,30,40,50,60,70,80]

labels = ['Age_Tier1','Age_Tier2','Age_Tier3','Age_Tier4','Age_Tier5','Age_Tier6']

combine['AGE'] = pd.cut(combine['AGE'], bins=bins, labels=labels, right=False)

combine['AGE'].value_counts()
combine['PAY_0'].value_counts()
combine['PAY_0'] = combine['PAY_0'].replace(0, "Paid_On_Time")

combine['PAY_0'] = combine['PAY_0'].replace(-1, "Duly_1")

combine['PAY_0'] = combine['PAY_0'].replace(1, "Delay_1")

combine['PAY_0'] = combine['PAY_0'].replace(-2, "Duly_2")

combine['PAY_0'] = combine['PAY_0'].replace(2, "Delay_2")

combine['PAY_0'] = combine['PAY_0'].replace(3, "Delay_3")

combine['PAY_0'] = combine['PAY_0'].replace(4, "Delay_4")

combine['PAY_0'] = combine['PAY_0'].replace(5, "Delay_5")

combine['PAY_0'] = combine['PAY_0'].replace(8, "Delay_8")

combine['PAY_0'] = combine['PAY_0'].replace(6, "Delay_6")

combine['PAY_0'] = combine['PAY_0'].replace(7, "Delay_7")

combine['PAY_0'].value_counts()
combine['PAY_2'].value_counts()
combine['PAY_2'] = combine['PAY_2'].replace(0, "Paid_On_Time")

combine['PAY_2'] = combine['PAY_2'].replace(-1, "Duly_1")

combine['PAY_2'] = combine['PAY_2'].replace(1, "Delay_1")

combine['PAY_2'] = combine['PAY_2'].replace(-2, "Duly_2")

combine['PAY_2'] = combine['PAY_2'].replace(2, "Delay_2")

combine['PAY_2'] = combine['PAY_2'].replace(3, "Delay_3")

combine['PAY_2'] = combine['PAY_2'].replace(4, "Delay_4")

combine['PAY_2'] = combine['PAY_2'].replace(5, "Delay_5")

combine['PAY_2'] = combine['PAY_2'].replace(8, "Delay_8")

combine['PAY_2'] = combine['PAY_2'].replace(6, "Delay_6")

combine['PAY_2'] = combine['PAY_2'].replace(7, "Delay_7")

combine['PAY_2'].value_counts()
combine['PAY_3'].value_counts()
combine['PAY_3'] = combine['PAY_3'].replace(0, "Paid_On_Time")

combine['PAY_3'] = combine['PAY_3'].replace(-1, "Duly_1")

combine['PAY_3'] = combine['PAY_3'].replace(1, "Delay_1")

combine['PAY_3'] = combine['PAY_3'].replace(-2, "Duly_2")

combine['PAY_3'] = combine['PAY_3'].replace(2, "Delay_2")

combine['PAY_3'] = combine['PAY_3'].replace(3, "Delay_3")

combine['PAY_3'] = combine['PAY_3'].replace(4, "Delay_4")

combine['PAY_3'] = combine['PAY_3'].replace(5, "Delay_5")

combine['PAY_3'] = combine['PAY_3'].replace(8, "Delay_8")

combine['PAY_3'] = combine['PAY_3'].replace(6, "Delay_6")

combine['PAY_3'] = combine['PAY_3'].replace(7, "Delay_7")

combine['PAY_3'].value_counts()
combine['PAY_4'].value_counts()
combine['PAY_4'] = combine['PAY_4'].replace(0, "Paid_On_Time")

combine['PAY_4'] = combine['PAY_4'].replace(-1, "Duly_1")

combine['PAY_4'] = combine['PAY_4'].replace(1, "Delay_1")

combine['PAY_4'] = combine['PAY_4'].replace(-2, "Duly_2")

combine['PAY_4'] = combine['PAY_4'].replace(2, "Delay_2")

combine['PAY_4'] = combine['PAY_4'].replace(3, "Delay_3")

combine['PAY_4'] = combine['PAY_4'].replace(4, "Delay_4")

combine['PAY_4'] = combine['PAY_4'].replace(5, "Delay_5")

combine['PAY_4'] = combine['PAY_4'].replace(8, "Delay_8")

combine['PAY_4'] = combine['PAY_4'].replace(6, "Delay_6")

combine['PAY_4'] = combine['PAY_4'].replace(7, "Delay_7")

combine['PAY_4'].value_counts()
combine['PAY_5'].value_counts()
combine['PAY_5'] = combine['PAY_5'].replace(0, "Paid_On_Time")

combine['PAY_5'] = combine['PAY_5'].replace(-1, "Duly_1")

combine['PAY_5'] = combine['PAY_5'].replace(1, "Delay_1")

combine['PAY_5'] = combine['PAY_5'].replace(-2, "Duly_2")

combine['PAY_5'] = combine['PAY_5'].replace(2, "Delay_2")

combine['PAY_5'] = combine['PAY_5'].replace(3, "Delay_3")

combine['PAY_5'] = combine['PAY_5'].replace(4, "Delay_4")

combine['PAY_5'] = combine['PAY_5'].replace(5, "Delay_5")

combine['PAY_5'] = combine['PAY_5'].replace(8, "Delay_8")

combine['PAY_5'] = combine['PAY_5'].replace(6, "Delay_6")

combine['PAY_5'] = combine['PAY_5'].replace(7, "Delay_7")

combine['PAY_5'].value_counts()
combine['PAY_6'].value_counts()
combine['PAY_6'] = combine['PAY_6'].replace(0, "Paid_On_Time")

combine['PAY_6'] = combine['PAY_6'].replace(-1, "Duly_1")

combine['PAY_6'] = combine['PAY_6'].replace(1, "Delay_1")

combine['PAY_6'] = combine['PAY_6'].replace(-2, "Duly_2")

combine['PAY_6'] = combine['PAY_6'].replace(2, "Delay_2")

combine['PAY_6'] = combine['PAY_6'].replace(3, "Delay_3")

combine['PAY_6'] = combine['PAY_6'].replace(4, "Delay_4")

combine['PAY_6'] = combine['PAY_6'].replace(5, "Delay_5")

combine['PAY_6'] = combine['PAY_6'].replace(8, "Delay_8")

combine['PAY_6'] = combine['PAY_6'].replace(6, "Delay_6")

combine['PAY_6'] = combine['PAY_6'].replace(7, "Delay_7")

combine['PAY_6'].value_counts()
bill_amount = ['BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6']

combine[bill_amount].describe()
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()



combine[bill_amount] = scaler.fit_transform(combine[bill_amount])

combine[bill_amount].describe()
pay_amount = ['PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6']

combine[pay_amount].describe()
combine[pay_amount] = scaler.fit_transform(combine[pay_amount])

combine[pay_amount].describe()
combine.head()
combine = pd.get_dummies(combine)

combine.shape
X = combine[combine['default_payment_next_month'].isnull()!=True].drop(['ID','default_payment_next_month'], axis=1)

y = combine[combine['default_payment_next_month'].isnull()!=True]['default_payment_next_month']



X_test = combine[combine['default_payment_next_month'].isnull()==True].drop(['ID','default_payment_next_month'], axis=1)



X.shape, y.shape, X_test.shape
from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=3,

            max_features=None, max_leaf_nodes=20,

            min_impurity_decrease=0.0, min_impurity_split=None,

            min_samples_leaf=1, min_samples_split=20,

            min_weight_fraction_leaf=0.0, presort=False, random_state=None,

            splitter='best')

model.fit(x_train,y_train)



pred_y = model.predict_proba(x_val)[:,1]
from lightgbm import LGBMClassifier

model = LGBMClassifier(n_estimator=5000,

                       random_state=1994,

                       learning_rate=0.05,

                       reg_alpha=0.2,

                       colsample_bytree=0.5,

                       bagging_fraction=0.9)



model.fit(x_train,y_train,

          eval_set=[(x_train,y_train),(x_val, y_val.values)],

          eval_metric='auc',

          early_stopping_rounds=100,

          verbose=200)



pred_y = model.predict_proba(x_val)[:,1]
from sklearn.metrics import accuracy_score,confusion_matrix,roc_auc_score

print(roc_auc_score(y_val, pred_y))

confusion_matrix(y_val,pred_y>0.5)
err = []

y_pred_tot_lgm = []



from sklearn.model_selection import StratifiedKFold



fold = StratifiedKFold(n_splits=10,shuffle=True,random_state=1994)

i = 1

for train_index, test_index in fold.split(X, y):

    x_train, x_val = X.iloc[train_index], X.iloc[test_index]

    y_train, y_val = y[train_index], y[test_index]

    m = LGBMClassifier(boosting_type='gbdt',

                       max_depth=5,

                       learning_rate=0.08,

                       n_estimators=5000,

                       min_child_weight=0.01,

                       colsample_bytree=0.5,

                       random_state=1994)

    m.fit(x_train, y_train,

          eval_set=[(x_train,y_train),(x_val, y_val)],

          early_stopping_rounds=200,

          eval_metric='auc',

          verbose=200)

    pred_y = m.predict_proba(x_val)[:,1]

    print("err_lgm: ",roc_auc_score(y_val,pred_y))

    err.append(roc_auc_score(y_val, pred_y))

    pred_test = m.predict_proba(X_test)[:,1]

    i = i + 1

    y_pred_tot_lgm.append(pred_test)
np.mean(err,0)
from xgboost import XGBClassifier



errxgb = []

y_pred_tot_xgb = []



from sklearn.model_selection import KFold,StratifiedKFold



fold = StratifiedKFold(n_splits=10,shuffle=True,random_state=1994)

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

          eval_metric='auc',

          verbose=200)

    pred_y = m.predict_proba(x_val)[:,-1]

    print("err_xgb: ",roc_auc_score(y_val,pred_y))

    errxgb.append(roc_auc_score(y_val, pred_y))

    pred_test = m.predict_proba(X_test)[:,-1]

    i = i + 1

    y_pred_tot_xgb.append(pred_test)
from catboost import CatBoostClassifier,Pool, cv

errCB = []

y_pred_tot_cb = []

from sklearn.model_selection import KFold,StratifiedKFold



fold = StratifiedKFold(n_splits=15,shuffle=True,random_state=1994)

i = 1

for train_index, test_index in fold.split(X,y):

    x_train, x_val = X.iloc[train_index], X.iloc[test_index]

    y_train, y_val = y[train_index], y[test_index]

    m = CatBoostClassifier(n_estimators=5000,

                           random_state=1994,

                           eval_metric='AUC',

                           learning_rate=0.03)

    m.fit(x_train, y_train,

          eval_set=[(x_train,y_train),(x_val, y_val)],

          early_stopping_rounds=200,

          verbose=200)

    pred_y = m.predict_proba(x_val)[:,-1]

    print("err_cb: ",roc_auc_score(y_val,pred_y))

    errCB.append(roc_auc_score(y_val,pred_y))

    pred_test = m.predict_proba(X_test)[:,-1]

    i = i + 1

    y_pred_tot_cb.append(pred_test)
np.mean(errCB)
(np.mean(errxgb, 0) + np.mean(err, 0) + np.mean(errCB, 0))/3
submission = pd.DataFrame()

submission['ID'] = combine[combine['default_payment_next_month'].isnull()==True]['ID']

submission['default_payment_next_month'] = (np.mean(y_pred_tot_lgm, 0) + np.mean(y_pred_tot_cb, 0) + 

                                            np.mean(y_pred_tot_xgb, 0))/3

submission.to_csv('rfr_lrg.csv', index=False, header=True)

submission.shape