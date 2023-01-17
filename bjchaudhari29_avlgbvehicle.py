import pandas as pd

import numpy as np

import datetime, time

from datetime import datetime

from time import mktime
data = pd.read_csv("../input/janatahack-machine-learning-for-iot-dataset/train_aWnotuB.csv")

submit = pd.read_csv("../input/janatahack-machine-learning-for-iot-dataset/test_BdBKkAj_L87Nc3S.csv")
submit.head()
submit['Vehicles'] = None
all_data = data.append(submit, ignore_index = True)
all_data.head()
all_data['DateTime']= pd.to_datetime(all_data['DateTime'])
all_data['month'] = all_data['DateTime'].dt.month
all_data['day'] = all_data['DateTime'].dt.day
all_data['year'] = all_data['DateTime'].dt.year
all_data['quarter'] = all_data['DateTime'].dt.quarter
all_data['dayofweek'] = all_data['DateTime'].dt.dayofweek
all_data['is_weekend'] = np.where(all_data['dayofweek'].isin([5,6]),1,0)
all_data['hour'] = all_data['DateTime'].dt.hour
all_data['is_night'] = np.where(all_data['hour'].isin([21,22,23,0,1,2,3,4,5]),1,0)
all_data['is_morning'] = np.where(all_data['hour'].isin([6,7,8,9,10]),1,0)
all_data['is_afternoon'] = np.where(all_data['hour'].isin([11,12,13,14,15,16]),1,0)
all_data['is_evening'] = np.where(all_data['hour'].isin([17,18,19,20]),1,0)
df_train=all_data[all_data['Vehicles'].isnull()==False].copy()

df_test=all_data[all_data['Vehicles'].isnull()==True].copy()
test_1 = df_test[df_test['Junction']== 1]

test_2 = df_test[df_test['Junction']== 2]

test_3 = df_test[df_test['Junction']== 3]

test_4 = df_test[df_test['Junction']== 4]

print(test_1.shape, test_2.shape,test_3.shape,test_4.shape)
train_1 = df_train[df_train['Junction']== 1]

train_2 = df_train[df_train['Junction']== 2]

train_3 = df_train[df_train['Junction']== 3]

train_4 = df_train[df_train['Junction']== 4]

print(train_1.shape, train_2.shape,train_3.shape,train_4.shape)
from sklearn.model_selection import StratifiedKFold,train_test_split

X_1,y_1=train_1.drop(['DateTime', 'ID', 'Junction','Vehicles'],axis=1),train_1['Vehicles']

Xtest_1=test_1.drop(['DateTime','ID', 'Junction','Vehicles'],axis=1)

y_1 = y_1.astype(int)

#X = X.fillna(0)

#Xtest = Xtest.fillna(0)

print(X_1.shape,Xtest_1.shape)

X_train_1,X_val_1,y_train_1,y_val_1 = train_test_split(X_1,y_1,test_size=0.20,random_state = 1996)
import lightgbm as lgb

from sklearn.metrics import mean_squared_error



lgb_train = lgb.Dataset(X_train_1, y_train_1)

lgb_eval = lgb.Dataset(X_val_1, y_val_1, reference=lgb_train)



params = {

    'boosting_type': 'gbdt',

    'objective': 'regression',

    'metric': {'l1', 'rmse'},

    'num_leaves': 31,

    'learning_rate': 0.05,

    'feature_fraction': 0.9,

    'bagging_fraction': 0.8,

    'bagging_freq': 5,

    'verbose': 0

}



gbm_1 = lgb.train(params,

                lgb_train,

                num_boost_round=500,

                valid_sets=[lgb_eval, lgb_train],

                early_stopping_rounds=5, verbose_eval = 10)



y_pred = gbm_1.predict(X_val_1, num_iteration=gbm_1.best_iteration)

# eval

print('The rmse of prediction is:', mean_squared_error(y_val_1, y_pred) ** 0.5)
from sklearn.model_selection import StratifiedKFold,train_test_split

X_2,y_2=train_2.drop(['DateTime', 'ID', 'Junction','Vehicles'],axis=1),train_2['Vehicles']

Xtest_2=test_2.drop(['DateTime','ID', 'Junction','Vehicles'],axis=1)

y_2 = y_2.astype(int)

#X = X.fillna(0)

#Xtest = Xtest.fillna(0)

print(X_2.shape,Xtest_2.shape)

X_train_2,X_val_2,y_train_2,y_val_2 = train_test_split(X_2,y_2,test_size=0.20,random_state = 1996)
import lightgbm as lgb

from sklearn.metrics import mean_squared_error



lgb_train = lgb.Dataset(X_train_2, y_train_2)

lgb_eval = lgb.Dataset(X_val_2, y_val_2, reference=lgb_train)



params = {

    'boosting_type': 'gbdt',

    'objective': 'regression',

    'metric': {'rmse', 'l1'},

    'num_leaves': 31,

    'learning_rate': 0.05,

    'feature_fraction': 0.9,

    'bagging_fraction': 0.8,

    'bagging_freq': 5,

    'verbose': 0

}



gbm_2 = lgb.train(params,

                lgb_train,

                num_boost_round=500,

                valid_sets=lgb_eval,

                early_stopping_rounds=5)



y_pred = gbm_2.predict(X_val_2, num_iteration=gbm_2.best_iteration)

# eval

print('The rmse of prediction is:', mean_squared_error(y_val_2, y_pred) ** 0.5)
from sklearn.model_selection import StratifiedKFold,train_test_split

X_3,y_3=train_3.drop(['DateTime', 'ID', 'Junction','Vehicles'],axis=1),train_3['Vehicles']

Xtest_3=test_3.drop(['DateTime','ID', 'Junction','Vehicles'],axis=1)

y_3 = y_3.astype(int)

#X = X.fillna(0)

#Xtest = Xtest.fillna(0)

print(X_3.shape,Xtest_3.shape)

X_train_3,X_val_3,y_train_3,y_val_3 = train_test_split(X_3,y_3,test_size=0.20,random_state = 1996)
import lightgbm as lgb

from sklearn.metrics import mean_squared_error



lgb_train = lgb.Dataset(X_train_3, y_train_3)

lgb_eval = lgb.Dataset(X_val_3, y_val_3, reference=lgb_train)



params = {

    'boosting_type': 'gbdt',

    'objective': 'regression',

    'metric': {'rmse', 'l1'},

    'num_leaves': 31,

    'learning_rate': 0.05,

    'feature_fraction': 0.9,

    'bagging_fraction': 0.8,

    'bagging_freq': 5,

    'verbose': 0

}



gbm_3 = lgb.train(params,

                lgb_train,

                num_boost_round=500,

                valid_sets=lgb_eval,

                early_stopping_rounds=5, verbose_eval = 100)



y_pred = gbm_3.predict(X_val_3, num_iteration=gbm_3.best_iteration)

# eval

print('The rmse of prediction is:', mean_squared_error(y_val_3, y_pred) ** 0.5)
from sklearn.model_selection import StratifiedKFold,train_test_split

X_4,y_4=train_4.drop(['DateTime', 'ID', 'Junction','Vehicles'],axis=1),train_4['Vehicles']

Xtest_4=test_4.drop(['DateTime','ID', 'Junction','Vehicles'],axis=1)

y_4 = y_4.astype(int)

#X = X.fillna(0)

#Xtest = Xtest.fillna(0)

print(X_4.shape,Xtest_4.shape)

X_train_4,X_val_4,y_train_4,y_val_4 = train_test_split(X_4,y_4,test_size=0.20,random_state = 1996)
import lightgbm as lgb

from sklearn.metrics import mean_squared_error



lgb_train = lgb.Dataset(X_train_4, y_train_4)

lgb_eval = lgb.Dataset(X_val_4, y_val_4, reference=lgb_train)



params = {

    'boosting_type': 'gbdt',

    'objective': 'regression',

    'metric': {'l2', 'l1'},

    'num_leaves': 31,

    'learning_rate': 0.05,

    'feature_fraction': 0.9,

    'bagging_fraction': 0.8,

    'bagging_freq': 5,

    'verbose': 0

}



gbm_4 = lgb.train(params,

                lgb_train,

                num_boost_round=500,

                valid_sets=lgb_eval,

                early_stopping_rounds=5, verbose_eval = 10)



y_pred = gbm_4.predict(X_val_4, num_iteration=gbm_4.best_iteration)

# eval

print('The rmse of prediction is:', mean_squared_error(y_val_4, y_pred) ** 0.5)
y_pred_1 = gbm_1.predict(Xtest_1, num_iteration=gbm_1.best_iteration)

y_pred_2 = gbm_2.predict(Xtest_2, num_iteration=gbm_2.best_iteration)

y_pred_3 = gbm_3.predict(Xtest_3, num_iteration=gbm_3.best_iteration)

y_pred_4 = gbm_4.predict(Xtest_4, num_iteration=gbm_4.best_iteration)
y_pred_1 = y_pred_1.astype(int)

y_pred_2 = y_pred_2.astype(int)

y_pred_3 = y_pred_3.astype(int)

y_pred_4 = y_pred_4.astype(int)
submission = []

for i in y_pred_1:

    submission.append(i)

for i in y_pred_2:

    submission.append(i)

for i in y_pred_3:

    submission.append(i)

for i in y_pred_4:

    submission.append(i)
len(submission)
forecast = pd.DataFrame(submit['ID'],columns=['ID'])
forecast['Vehicles'] = submission
forecast.to_csv('Submission.csv', index = False)