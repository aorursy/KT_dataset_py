# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
ts = pd.read_csv('/kaggle/input/summeranalytics2020/train.csv')
pd.set_option('display.max_columns',None)
ts
test_set = pd.read_csv('/kaggle/input/summeranalytics2020/test.csv')
test_set.columns
ts.columns
display(ts.isnull().any())
ts.dtypes
ts.columns.nunique()
from matplotlib import pyplot as plt
import seaborn as sns

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

plt.figure(figsize=(40,24))

numerical = [u'Age', u'DistanceFromHome', 
             u'Education', u'EmployeeNumber', u'EnvironmentSatisfaction',
             u'JobInvolvement', u'JobSatisfaction',
             u'MonthlyIncome', u'NumCompaniesWorked',
             u'PercentSalaryHike', u'PerformanceRating',
             u'StockOptionLevel', u'TotalWorkingYears',
             u'TrainingTimesLastYear', u'YearsAtCompany',
             u'YearsInCurrentRole', u'YearsSinceLastPromotion',u'YearsWithCurrManager', u'CommunicationSkill', u'Behaviour']

corr = ts[numerical].corr()

sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0
    ,square=True, cmap="BuPu")
ts1 = ts.drop(['Attrition'], axis=1)


categorical = []
for col, value in ts1.iteritems():
    if value.dtype == 'object':
        categorical.append(col)

numerical = ts1.columns.difference(categorical)
ts1_cat = ts1[categorical]

ts1_cat = pd.get_dummies(ts1_cat)
ts1_cat.head(3)
ts1_num = ts1[numerical]
ts1_final = pd.concat([ts1_num, ts1_cat], axis=1)

sns.barplot( x=ts["Attrition"].value_counts().index.values,
            y= ts["Attrition"].value_counts().values)



from sklearn.model_selection import train_test_split
target = ts['Attrition']
x_train, x_val, y_train, y_val = train_test_split(ts1_final, 
                                                          target,
                                                         train_size= 0.80,
                                                         random_state=0)
print(x_train.shape)
print(x_val.shape)
print(y_train.shape)
print(y_val.shape)
#Logistic REGRESSION MODEL
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

lg = LogisticRegression()
lg.fit(x_train, y_train)

lpg = lg.predict_proba(x_val)[:,1]

print(lpg)


len(lpg)
test_set = pd.read_csv('/kaggle/input/summeranalytics2020/test.csv')
test_set
categorical_test_set = []
for col, value in test_set.iteritems():
    if value.dtype == 'object':
        categorical_test_set.append(col)

numerical_test_set = test_set.columns.difference(categorical_test_set)
test_set_cat = test_set[categorical_test_set]
test_set_cat = pd.get_dummies(test_set_cat)
test_set_cat.head(3)
test_set_num = test_set[numerical_test_set]
test_set_final = pd.concat([test_set_num, test_set_cat], axis=1)
lpg_test_set = lg.predict_proba(test_set_final)[:,1]

print(lpg_test_set)
submission_1 = {'Id': test_set['Id'],
               'Attrition': lpg_test_set}
sb_1 = pd.DataFrame(submission_1, columns = ['Id', 'Attrition'])
sb_1
sb_1.reset_index(drop=True, inplace=True)
pd.set_option('display.max_rows',500)
sb_1
sb_1.to_csv(r'C:\Users\arshk\sb1.csv', index = False, header=True)
seed = 0   

rf_params = {
    'n_jobs': -1,
    'n_estimators': 1000, 
    'max_features': 0.3,
    'max_depth': 4,
    'min_samples_leaf': 2,
    'max_features' : 'sqrt',
    'random_state' : seed,
    'verbose': 0
}
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(**rf_params)
rf.fit(x_train, y_train)
rf_val_pred = rf.predict(x_val)
ras = roc_auc_score(y_val, rf_val_pred)
print(ras)
rf_test_set = rf.predict_proba(test_set_final)[:,1]

print(rf_test_set)
submission_2 = {'Id': test_set['Id'],
               'Attrition': rf_test_set}
sb_2 = pd.DataFrame(submission_2, columns = ['Id', 'Attrition'])
sb_2.head(500)
gb_params ={
    'n_estimators': 1500,
    'max_features': 0.9,
    'learning_rate' : 0.25,
    'max_depth': 4,
    'min_samples_leaf': 2,
    'subsample': 1,
    'max_features' : 'sqrt',
    'random_state' : seed,
    'verbose': 0
}
from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(**gb_params)

gb.fit(x_train, y_train)


gb_val_pred = gb.predict_proba(x_val)[:,1]
ras_gb = roc_auc_score(y_val, gb_val_pred)
print(ras_gb)
gb_test_set = gb.predict_proba(test_set_final)[:,1]

print(gb_test_set)
submission_3 = {'Id': test_set['Id'],
               'Attrition': gb_test_set}
sb_3 = pd.DataFrame(submission_3, columns = ['Id', 'Attrition'])
sb_3
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

lg = LogisticRegression(solver='lbfgs', max_iter=2000, C=0.5, penalty='l2', random_state=1)
lg.fit(x_train, y_train)

lpg_2 = lg.predict_proba(test_set_final)[:,1]

print(lpg_2)

submission_4 = {'Id': test_set['Id'],
               'Attrition': lpg_2}
sb_4 = pd.DataFrame(submission_4, columns = ['Id', 'Attrition'])
sb_4
seed = 0   

rf_params_2 = {
    'n_jobs': -1,
    'n_estimators': 1000, 
    'max_features': 0.3,
    'max_depth': 4,
    'min_samples_leaf': 2,
    'max_features' : 'sqrt',
    'random_state' : seed,
    'verbose': 0
}

from sklearn.ensemble import RandomForestClassifier

rf_2 = RandomForestClassifier(**rf_params_2)
rf_2.fit(x_train, y_train)

rf_val_pred_2 = rf_2.predict(x_val)

ras_2 = roc_auc_score(y_val, rf_val_pred_2)
print(ras_2)


rf_2_test_set = rf_2.predict_proba(test_set_final)[:,1]

print(rf_2_test_set)
submission_5 = {'Id': test_set['Id'],
               'Attrition': rf_2_test_set}
sb_5 = pd.DataFrame(submission_5, columns = ['Id', 'Attrition'])
sb_5
from sklearn.svm import SVC

svc = SVC(kernel='rbf', max_iter = 1000, random_state=1, gamma='scale',probability=True)
svc_tr = svc.fit(x_train, y_train)


svc_val = svc_tr.predict(x_val)

ras_3 = roc_auc_score(y_val, svc_val)
print(ras_3)

svc_test_set = svc_tr.predict_proba(test_set_final)[:,1]

print(svc_test_set)
submission_6 = {'Id': test_set['Id'],
               'Attrition': svc_test_set}
sb_6 = pd.DataFrame(submission_6, columns = ['Id', 'Attrition'])
sb_6
seed = 0   

rf_params_3 = {
    'n_jobs': -1,
    'n_estimators': 1000, 
    'max_features': 0.3,
    'max_depth': 6,
    'min_samples_leaf': 4,
    'max_features' : 'sqrt',
    'random_state' : seed,
    'verbose': 0
}

from sklearn.ensemble import RandomForestClassifier

rf_3 = RandomForestClassifier(**rf_params_3)
rf_3.fit(x_train, y_train)

rf_val_pred_3 = rf_3.predict(x_val)

ras_3 = roc_auc_score(y_val, rf_val_pred_3)
print(ras_3)

rf_3_test_set = rf_3.predict_proba(test_set_final)[:,1]

print(rf_3_test_set)
submission_7 = {'Id': test_set['Id'],
               'Attrition': rf_3_test_set}
sb_7 = pd.DataFrame(submission_7, columns = ['Id', 'Attrition'])
sb_7
gb_params_2 ={
    'n_estimators': 1500,
    'max_features': 0.9,
    'learning_rate' : 0.25,
    'max_depth': 4,
    'min_samples_leaf': 4,
    'subsample': 1,
    'max_features' : 'sqrt',
    'random_state' : seed,
    'verbose': 0
}

from sklearn.ensemble import GradientBoostingClassifier

gb_2 = GradientBoostingClassifier(**gb_params_2)
gb_2.fit(x_train, y_train)


gb_val_pred_2 = gb_2.predict_proba(x_val)[:,1]

ras_gb_2 = roc_auc_score(y_val, gb_val_pred_2)
print(ras_gb_2)
gb_test_set_2 = gb_2.predict_proba(test_set_final)[:,1]

print(gb_test_set_2)
submission_9 = {'Id': test_set['Id'],
               'Attrition': gb_test_set_2}
sb_9 = pd.DataFrame(submission_9, columns = ['Id', 'Attrition'])
sb_9
gb_params_3 ={
    'n_estimators': 1500,
    'max_features': 0.9,
    'learning_rate' : 0.25,
    'max_depth': 6,
    'min_samples_leaf': 2,
    'subsample': 1,
    'max_features' : 'sqrt',
    'random_state' : seed,
    'verbose': 0
}

from sklearn.ensemble import GradientBoostingClassifier

gb_3 = GradientBoostingClassifier(**gb_params_3)
gb_3.fit(x_train, y_train)


gb_val_pred_3 = gb_3.predict_proba(x_val)[:,1]

ras_gb_3 = roc_auc_score(y_val, gb_val_pred_3)
print(ras_gb_3)
gb_test_set_3 = gb_3.predict_proba(test_set_final)[:,1]

submission_10 = {'Id': test_set['Id'],
               'Attrition': gb_test_set_3}
sb_10 = pd.DataFrame(submission_10, columns = ['Id', 'Attrition'])

sb_10
gb_params_4 ={
    'n_estimators': 1500,
    'max_features': 0.9,
    'learning_rate' : 0.20,
    'max_depth': 6,
    'min_samples_leaf': 3,
    'subsample': 1,
    'max_features' : 'sqrt',
    'random_state' : seed,
    'verbose': 0
}

from sklearn.ensemble import GradientBoostingClassifier

gb_4 = GradientBoostingClassifier(**gb_params_4)
gb_4.fit(x_train, y_train)


gb_val_pred_4 = gb_4.predict_proba(x_val)[:,1]

ras_gb_4 = roc_auc_score(y_val, gb_val_pred_4)
print(ras_gb_4)
gb_test_set_4 = gb_4.predict_proba(test_set_final)[:,1]

submission_11 = {'Id': test_set['Id'],
               'Attrition': gb_test_set_4}
sb_11 = pd.DataFrame(submission_11, columns = ['Id', 'Attrition'])

sb_11
gb_params_5 ={
    'n_estimators': 1500,
    'max_features': 0.9,
    'learning_rate' : 0.20,
    'max_depth': 6,
    'min_samples_leaf': 2,
    'subsample': 1,
    'max_features' : 'sqrt',
    'random_state' : seed,
    'verbose': 0
}

from sklearn.ensemble import GradientBoostingClassifier

gb_5 = GradientBoostingClassifier(**gb_params_5)
gb_5.fit(x_train, y_train)


gb_val_pred_5 = gb_5.predict_proba(x_val)[:,1]

ras_gb_5 = roc_auc_score(y_val, gb_val_pred_5)
print(ras_gb_5)
gb_test_set_5 = gb_5.predict_proba(test_set_final)[:,1]

submission_14 = {'Id': test_set['Id'],
               'Attrition': gb_test_set_5}
sb_14 = pd.DataFrame(submission_14, columns = ['Id', 'Attrition'])

sb_14 
gb_params_6 ={
    'n_estimators': 1500,
    'max_features': 0.9,
    'learning_rate' : 0.30,
    'max_depth': 6,
    'min_samples_leaf': 2,
    'subsample': 1,
    'max_features' : 'sqrt',
    'random_state' : seed,
    'verbose': 0
}

from sklearn.ensemble import GradientBoostingClassifier

gb_6 = GradientBoostingClassifier(**gb_params_6)
gb_6.fit(x_train, y_train)


gb_val_pred_6 = gb_6.predict_proba(x_val)[:,1]

ras_gb_6 = roc_auc_score(y_val, gb_val_pred_6)
print(ras_gb_6)
gb_test_set_6 = gb_6.predict_proba(test_set_final)[:,1]

submission_15 = {'Id': test_set['Id'],
               'Attrition': gb_test_set_6}
sb_15 = pd.DataFrame(submission_15, columns = ['Id', 'Attrition'])

sb_15 
import xgboost as xgb
data_Dmatrix = xgb.DMatrix(data=ts1_final, label=target)
xgb_clf = xgb.XGBClassifier(objective='binary:logistic',
                            n_estimators = 1000,
                            max_depth = 6,
                            colsample_bytree = 0.3,
                            learning_rate = 0.35,
                            alpha = 6,
                            verbose = 0,
                            n_jobs = -1)

xgc = xgb_clf.fit(x_train,y_train)

xgc_predict = xgc.predict_proba(x_val)[:,1]

rac_x = roc_auc_score(y_val, xgc_predict)
print(rac_x)
data_Dmatrix_test = xgb.DMatrix(data = test_set_final)
xgb_test_set = xgc.predict_proba(test_set_final)[:,1]

submission_17 = {'Id': test_set['Id'],
               'Attrition': xgb_test_set}
sb_17 = pd.DataFrame(submission_17, columns = ['Id', 'Attrition'])

sb_17 
xgb_clf_2 = xgb.XGBClassifier(objective='binary:logistic',
                            n_estimators = 1000,
                            max_depth = 5,
                            colsample_bytree = 0.2,
                            learning_rate = 0.25,
                            alpha = 5,
                            verbose = 0,
                            n_jobs = -1)

xgc_2 = xgb_clf_2.fit(x_train,y_train)

xgc_predict_2 = xgc_2.predict_proba(x_val)[:,1]

rac_x_2 = roc_auc_score(y_val, xgc_predict_2)
print(rac_x_2)
xgb_test_set_2 = xgc_2.predict_proba(test_set_final)[:,1]

submission_25 = {'Id': test_set['Id'],
               'Attrition': xgb_test_set_2}
sb_25 = pd.DataFrame(submission_25, columns = ['Id', 'Attrition'])

sb_25 
xgb_clf_3 = xgb.XGBClassifier(objective='binary:logistic',
                            n_estimators = 1000,
                            max_depth = 7,
                            colsample_bytree = 0.3,
                            learning_rate = 0.30,
                            alpha = 6,
                            verbose = 0,
                            n_jobs = -1)

xgc_3 = xgb_clf_3.fit(x_train,y_train)

xgc_predict_3 = xgc_3.predict_proba(x_val)[:,1]

rac_x_3 = roc_auc_score(y_val, xgc_predict_3)
print(rac_x_3)
xgb_test_set_3 = xgc_3.predict_proba(test_set_final)[:,1]

submission_28 = {'Id': test_set['Id'],
               'Attrition': xgb_test_set_3}
sb_28 = pd.DataFrame(submission_28, columns = ['Id', 'Attrition'])

sb_28 
xgb_clf_4 = xgb.XGBClassifier(objective='binary:logistic',
                            n_estimators = 1000,
                            max_depth = 5,
                            colsample_bytree = 0.3,
                            learning_rate = 0.25,
                            alpha = 5,
                            verbose = 0,
                            n_jobs = -1)

xgc_4 = xgb_clf_4.fit(x_train,y_train)

xgc_predict_4 = xgc_4.predict_proba(x_val)[:,1]

rac_x_4 = roc_auc_score(y_val, xgc_predict_4)
print(rac_x_4)
xgb_test_set_4 = xgc_4.predict_proba(test_set_final)[:,1]

submission_35 = {'Id': test_set['Id'],
               'Attrition': xgb_test_set_4}
sb_35 = pd.DataFrame(submission_35, columns = ['Id', 'Attrition'])

sb_35
import lightgbm
from lightgbm import LGBMClassifier
lgbm_model = LGBMClassifier(n_jobs = -1,
                            boosting_type = 'gbdt', 
                           n_estimators = 1000,
                           learning_rate = 0.10,
                           colsample_bytree = 0.25,
                           reg_alpha = 1,
                           verbose = 0,
                           max_depth = 7,
                            
                           )

lgbm = lgbm_model.fit(x_train, y_train)

lgbm_predict = lgbm.predict_proba(x_val)[:,1]

ras_lg = roc_auc_score(y_val, lgbm_predict)
print(ras_lg)
lgbm_test_set = lgbm.predict_proba(test_set_final)[:,1]

submission_45 = {'Id': test_set['Id'],
               'Attrition': lgbm_test_set}
sb_45 = pd.DataFrame(submission_45, columns = ['Id', 'Attrition'])

sb_45
ts1_final_2 = ts1_final.drop(['Id','EmployeeNumber'],axis=1)
x_train_2, x_val_2, y_train_2, y_val_2 = train_test_split(ts1_final_2, 
                                                          target,
                                                         train_size= 0.80,
                                                         random_state=0)
test_set_final_2 = test_set_final.drop(['Id','EmployeeNumber'],axis=1)
xgb_clf_5 = xgb.XGBClassifier(objective='binary:logistic',
                            n_estimators = 1000,
                            max_depth = 8,
                            colsample_bytree = 0.30,
                            learning_rate = 0.30,
                            
                            verbose = 0,
                            n_jobs = -1)

xgc_5 = xgb_clf_5.fit(x_train_2,y_train_2)

xgc_predict_5 = xgc_5.predict_proba(x_val_2)[:,1]

rac_x_5 = roc_auc_score(y_val_2, xgc_predict_5)
print(rac_x_5)
xgb_test_set_5 = xgc_5.predict_proba(test_set_final_2)[:,1]

submission_54 = {'Id': test_set['Id'],
               'Attrition': xgb_test_set_5}
sb_54 = pd.DataFrame(submission_54, columns = ['Id', 'Attrition'])

sb_54
xgb_clf_6 = xgb.XGBClassifier(objective='binary:logistic',
                            n_estimators = 1000,
                            max_depth = 9,
                            colsample_bytree = 0.3,
                            learning_rate = 0.30,
                            alpha = 5,
                            verbose = 0,
                            n_jobs = -1)

xgc_6 = xgb_clf_6.fit(x_train_2,y_train_2)

xgc_predict_6 = xgc_6.predict_proba(x_val_2)[:,1]

rac_x_6 = roc_auc_score(y_val_2, xgc_predict_6)
print(rac_x_6)
xgb_test_set_6 = xgc_6.predict_proba(test_set_final_2)[:,1]

submission_57 = {'Id': test_set['Id'],
               'Attrition': xgb_test_set_6}
sb_57 = pd.DataFrame(submission_57, columns = ['Id', 'Attrition'])

sb_57
gb_params_7 ={
    'n_estimators': 1500,
    'max_features': 0.9,
    'learning_rate' : 0.30,
    'max_depth': 5,
    'min_samples_leaf': 3,
    'subsample': 1,
    'max_features' : 'sqrt',
    'random_state' : seed,
    'verbose': 0
}

from sklearn.ensemble import GradientBoostingClassifier

gb_7 = GradientBoostingClassifier(**gb_params_7)
gb_7.fit(x_train_2, y_train_2)


gb_val_pred_7 = gb_7.predict_proba(x_val_2)[:,1]

ras_gb_7 = roc_auc_score(y_val_2, gb_val_pred_7)
print(ras_gb_7)
gb_test_set_7 = gb_7.predict_proba(test_set_final_2)[:,1]

submission_58 = {'Id': test_set['Id'],
               'Attrition': gb_test_set_7}
sb_58 = pd.DataFrame(submission_58, columns = ['Id', 'Attrition'])

sb_58
gb_params_8 ={
    'n_estimators': 1500,
    'max_features': 0.9,
    'learning_rate' : 0.30,
    'max_depth': 4,
    'min_samples_leaf': 3,
    'subsample': 1,
    'max_features' : 'sqrt',
    'random_state' : seed,
    'verbose': 0
}

from sklearn.ensemble import GradientBoostingClassifier

gb_8 = GradientBoostingClassifier(**gb_params_8)
gb_8.fit(x_train_2, y_train_2)


gb_val_pred_8 = gb_8.predict_proba(x_val_2)[:,1]

ras_gb_8 = roc_auc_score(y_val_2, gb_val_pred_8)
print(ras_gb_8)
gb_test_set_8 = gb_8.predict_proba(test_set_final_2)[:,1]

submission_64 = {'Id': test_set['Id'],
               'Attrition': gb_test_set_8}
sb_64 = pd.DataFrame(submission_64, columns = ['Id', 'Attrition'])

sb_64
x_train_3, x_val_3, y_train_3, y_val_3 = train_test_split(ts1_final_2, 
                                                          target,
                                                         train_size= 0.95,
                                                         random_state=0)
gb_params_9 ={
    'n_estimators': 1500,
    'max_features': 0.9,
    'learning_rate' : 0.30,
    'max_depth': 6,
    'min_samples_leaf': 2,
    'subsample': 1,
    'max_features' : 'sqrt',
    'random_state' : seed,
    'verbose': 0
}

from sklearn.ensemble import GradientBoostingClassifier

gb_9 = GradientBoostingClassifier(**gb_params_9)
gb_9.fit(x_train_3, y_train_3)


gb_val_pred_9 = gb_9.predict_proba(x_val_3)[:,1]

ras_gb_9 = roc_auc_score(y_val_3, gb_val_pred_9)
print(ras_gb_9)
gb_test_set_9 = gb_9.predict_proba(test_set_final_2)[:,1]

submission_71 = {'Id': test_set['Id'],
               'Attrition': gb_test_set_9}
sb_71 = pd.DataFrame(submission_71, columns = ['Id', 'Attrition'])

sb_71
xgb_clf_7 = xgb.XGBClassifier(objective='binary:logistic',
                            n_estimators = 1000,
                            max_depth = 6,
                            colsample_bytree = 0.3,
                            min_samples_leaf = 2,
                            learning_rate = 0.25,
                            alpha = 4,
                            verbose = 0,
                            n_jobs = -1)

xgc_7 = xgb_clf_7.fit(x_train_3,y_train_3)

xgc_predict_7 = xgc_7.predict_proba(x_val_3)[:,1]

rac_x_7 = roc_auc_score(y_val_3, xgc_predict_7)
print(rac_x_7)
xgb_test_set_7 = xgc_7.predict_proba(test_set_final_2)[:,1]

submission_82 = {'Id': test_set['Id'],
               'Attrition': xgb_test_set_7}
sb_82 = pd.DataFrame(submission_82, columns = ['Id', 'Attrition'])

sb_82
xgb_clf_8 = xgb.XGBClassifier(objective='binary:logistic',
                            n_estimators = 1000,
                            max_depth = 10,
                            eta = 0.01,
                            colsample_bytree = 1,
                            min_child_weight = 5,
                            subsample = 0.8,
                            learning_rate = 0.25,
                            alpha = 6,
                            verbose = 0,
                            n_jobs = -1)

xgc_8 = xgb_clf_8.fit(x_train_3,y_train_3)

xgc_predict_8 = xgc_8.predict_proba(x_val_3)[:,1]

rac_x_8 = roc_auc_score(y_val_3, xgc_predict_8)
print(rac_x_8)
xgb_test_set_8 = xgc_8.predict_proba(test_set_final_2)[:,1]

submission_96 = {'Id': test_set['Id'],
               'Attrition': xgb_test_set_8}
sb_96 = pd.DataFrame(submission_96, columns = ['Id', 'Attrition'])

sb_96
xgb_clf_9 = xgb.XGBClassifier(objective='binary:logistic',
                            n_estimators = 1000,
                            max_depth = 11,
                            eta = 0.01,
                            colsample_bytree = 1,
                            min_child_weight = 5,
                            subsample = 0.8,
                            learning_rate = 0.25,
                            alpha = 6,
                            verbose = 0,
                            n_jobs = -1)

xgc_9 = xgb_clf_9.fit(x_train_3,y_train_3)

xgc_predict_9 = xgc_9.predict_proba(x_val_3)[:,1]

rac_x_9 = roc_auc_score(y_val_3, xgc_predict_9)
print(rac_x_9)
xgb_test_set_9 = xgc_9.predict_proba(test_set_final_2)[:,1]

submission_97 = {'Id': test_set['Id'],
               'Attrition': xgb_test_set_9}
sb_97 = pd.DataFrame(submission_97, columns = ['Id', 'Attrition'])

sb_97
xgb_clf_10 = xgb.XGBClassifier(objective='binary:logistic',
                            n_estimators = 1000,
                            max_depth = 11,
                            eta = 0.01,
                            colsample_bytree = 1,
                            min_child_weight = 4,
                            subsample = 0.8,
                            learning_rate = 0.25,
                            alpha = 6,
                            verbose = 0,
                            n_jobs = -1)

xgc_10 = xgb_clf_10.fit(x_train_3,y_train_3)

xgc_predict_10 = xgc_10.predict_proba(x_val_3)[:,1]

rac_x_10 = roc_auc_score(y_val_3, xgc_predict_10)
print(rac_x_10)
xgb_test_set_10 = xgc_10.predict_proba(test_set_final_2)[:,1]

submission_102 = {'Id': test_set['Id'],
               'Attrition': xgb_test_set_10}
sb_102 = pd.DataFrame(submission_102, columns = ['Id', 'Attrition'])

sb_102
