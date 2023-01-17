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
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier


# loading data
df = pd.read_csv('/kaggle/input/porto-seguro-safe-driver-prediction/train.csv')
# Let's see about data.
df.info()
# let's see there are null values or not.
df.isnull().sum()
# make cat, bin, [con,or] in a list.
cat_feature = []
bin_feature = []
con_or_feature = []
for i in df.columns:
    if 'cat' in i:
        cat_feature.append(i)
    elif 'bin' in i:
        bin_feature.append(i)
    else:
        con_or_feature.append(i)
print(cat_feature)
print('cat_feature count:',len(cat_feature))
print(bin_feature)
print('bin_feature count:',len(bin_feature))
print(con_or_feature)
print('con_or_feature count:',len(con_or_feature))
# -1이 있는 변수만 보기
nulled_data = []
for i in cat_feature:
    if -1 in df[i].values:
        print(i)
        nulled_data.append(i)
for i in bin_feature:
    if -1 in df[i].values:
        print(i)
        nulled_data.append(i)
for i in con_or_feature:
    if -1 in df[i].values:
        print(i)
        nulled_data.append(i)
# count values
for i in nulled_data:
    print(df[i].value_counts())
# 13 features has null values 
print('ps_ind_02_cat: {:.5%}'.format(216/len(df)))
print('ps_ind_04_cat: {:.5%}'.format(83/len(df)))
print('ps_ind_05_cat: {:.5%}'.format(5809/len(df)))
print('ps_car_01_cat: {:.5%}'.format(107/len(df)))
print('ps_car_02_cat: {:.5%}'.format(5/len(df)))
print('ps_car_03_cat: {:.5%}'.format(411231/len(df)))
print('ps_car_05_cat: {:.5%}'.format(266551/len(df)))
print('ps_car_07_cat: {:.5%}'.format(11489/len(df)))
print('ps_car_09_cat: {:.5%}'.format(569/len(df)))
print('ps_reg_03: {:.5%}'.format(107772/len(df)))
print('ps_car_11: {:.5%}'.format(5/len(df)))
print('ps_car_12: {:.5%}'.format(1/len(df)))
print('ps_car_14: {:.5%}'.format(42620/len(df)))

df2 = df.copy()
df2.drop(['id','ps_car_03_cat','ps_car_05_cat','ps_reg_03','ps_car_14'], axis=1, inplace=True)
df2 = df2[(df2['ps_car_11'] != -1) & (df2['ps_car_12'] != -1) & (df2['ps_car_02_cat'] != -1)
          & (df2['ps_ind_02_cat'] != -1) & (df2['ps_ind_04_cat'] != -1) & (df2['ps_ind_05_cat'] != -1)
          & (df2['ps_car_01_cat'] != -1) & (df2['ps_car_07_cat'] != -1) & (df2['ps_car_09_cat'] != -1)]
for i in df.columns:
    if -1 in df[i].values:
        print(df[i].value_counts())

print('--'*20)
for i in df2.columns:
    if -1 in df2[i].values:
        print(df2[i].value_counts())


df2['target'].value_counts()
df2.columns

X,y = df2.iloc[:,1:], df2.iloc[:,:1]
print('Resampled dataset shape %s' % y['target'].value_counts())
# sm = SMOTE(random_state=42)
# X_res, y_res = sm.fit_resample(X, y)
rus = RandomUnderSampler(random_state=42)
X_res, y_res = rus.fit_resample(X,y)
print('Resampled dataset shape %s' % y_res['target'].value_counts())
# make cat, bin, [con,or] in a list.
cat_bin_feature = []
con_or_feature = []
for i in X_res.columns:
    if 'cat' in i:
        cat_bin_feature.append(i)
    elif 'bin' in i:
        cat_bin_feature.append(i)
    else:
        con_or_feature.append(i)

for i in cat_bin_feature:
    X_res[i] = X_res[i].astype('category')

X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size = 0.2, random_state=42)
X_train, X_test, y_train, y_test = X_train.values, X_test.values, y_train.values.ravel(), y_test.values.ravel()
# baseline = LogisticRegression()
# baseline.fit(X_train, y_train)
# print(baseline.score(X_train, y_train))
# print(baseline.score(X_test, y_test))
rf_param_grid = {'max_depth' : [6,7,8], 
                 'n_estimators' : [100,150,200],
                 'min_samples_split' : [2,3],
                 'min_samples_leaf' : [1,3,5]}


rf = RandomForestClassifier()
rf_grid = GridSearchCV(estimator=rf,
                       param_grid=rf_param_grid,
                       scoring='roc_auc',
                       cv=5)
rf_grid.fit(X_train, y_train)
best_max_depth = rf_grid.best_params_["max_depth"]
best_min_samples_split = rf_grid.best_params_["min_samples_split"]
best_n_estimators = rf_grid.best_params_['n_estimators']
best_min_samples_leaf = rf_grid.best_params_['min_samples_leaf']
#best_learning_rate = rf_grid.best_params_['learning_rate']

print('max_depth : ',best_max_depth,'\n',
     'min_samples_split : ',best_min_samples_split,'\n',
     'n_estimators : ',best_n_estimators,'\n')
     #'learning_rate : ',best_learning_rate,'\n')
rf_grid.best_params_
rf_2 = RandomForestClassifier(max_depth=8,
                            min_samples_leaf=1,
                            min_samples_split=3,
                            n_estimators=200)

rf_2.fit(X_train, y_train)
rf_2.score(X_test, y_test)
predict2 = rf_2.predict(submit_X)
#submission = pd.read_csv('/kaggle/input/porto-seguro-safe-driver-prediction/sample_submission.csv')
submission['target'] = predict2
submission['target'].value_counts()
submit_X = pd.read_csv('/kaggle/input/porto-seguro-safe-driver-prediction/test.csv')
submit_X.drop(['id','ps_car_03_cat','ps_car_05_cat','ps_reg_03','ps_car_14'], axis=1, inplace=True)
for i in cat_bin_feature:
    submit_X[i] = submit_X[i].astype('category')
submit_X = submit_X.values

submission = pd.read_csv('/kaggle/input/porto-seguro-safe-driver-prediction/sample_submission.csv')
submission['target'] = predict2
print(submission['target'].value_counts())

submission.to_csv('gridsearch_rf_0407_2_undersampling.csv', index=False)
print('저장하였습니다.')
predict = baseline.predict(submit_X)
submission = pd.read_csv('/kaggle/input/porto-seguro-safe-driver-prediction/sample_submission.csv')
submission['target'] = predict
import lightgbm as lgb
# from lightgbm import LGBMClassifier
# params = {'learning_rate': 0.01,
#           'max_depth': 16,
#           'boosting': 'gbdt',
#           'objective': 'binary',
#           'metric': 'auc',
#           'is_training_metric': True, 
#           'num_leaves': 144,
#           'feature_fraction': 0.9,
#           'bagging_fraction': 0.7, 
#           'bagging_freq': 5, 
#           'seed':2018} 

# train_ds = lgb.Dataset(X_train, label = y_train)
# val_ds = lgb.Dataset(X_test, label = y_test)

# model = lgb.train(params, train_ds, 1000, val_ds, verbose_eval=10, early_stopping_rounds=100)

# lgb_cf = LGBMClassifier()
# lgb_cf.fit(X_train,y_train)
# print(lgb_cf.score(X_test,y_test))


param_grid = {
    'num_leaves': [31, 127],
    'reg_alpha': [0.1, 0.5],
    'min_data_in_leaf': [30, 50, 100, 300, 400],
    'lambda_l1': [0, 1, 1.5],
    'lambda_l2': [0, 1]
    }

lgb_estimator = lgb.LGBMClassifier(boosting_type='gbdt',  objective='binary', num_boost_round=2000, learning_rate=0.01, metric='auc')

gsearch = GridSearchCV(estimator=lgb_estimator, param_grid=param_grid, cv=5)
lgb_model = gsearch.fit(X=X_train, y=y_train)

print(lgb_model.best_params_, lgb_model.best_score_)
predict_lgb = lgb_cf.predict(submit_X)
submission['target'] = predict_lgb
#submission['target'] = submission['target'].apply(lambda x: 1 if x>=0.5  else 0)
print(submission['target'].value_counts())

submission.to_csv('lgb_0407_1_undersampling.csv', index=False)
print('저장하였습니다.')

submission.to_csv('lgb_0407_2_undersampling.csv', index=False)
