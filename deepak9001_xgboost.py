# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import xgboost as xgb

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from sklearn.metrics import f1_score



train=pd.read_csv('../input/train.csv')

test=pd.read_csv('../input/test.csv')

print(train.shape)

print(test.shape)

train.columns
train[train<0]=0 

test[test<0]=0 
Y = train['target'].values

id_train = train['id'].values

id_test = test['id'].values
cat_cols=['ps_ind_02_cat','ps_ind_04_cat','ps_ind_05_cat','ps_car_01_cat','ps_car_02_cat',

       'ps_car_03_cat', 'ps_car_04_cat', 'ps_car_05_cat', 'ps_car_06_cat',

       'ps_car_07_cat', 'ps_car_08_cat', 'ps_car_09_cat', 'ps_car_10_cat',

       'ps_car_11_cat','ps_ind_06_bin', 'ps_ind_07_bin','ps_ind_08_bin', 'ps_ind_09_bin', 'ps_ind_10_bin', 

          'ps_ind_11_bin','ps_ind_12_bin', 'ps_ind_13_bin','ps_ind_16_bin', 'ps_ind_17_bin', 

          'ps_ind_18_bin','ps_calc_15_bin', 'ps_calc_16_bin', 'ps_calc_17_bin', 'ps_calc_18_bin',

       'ps_calc_19_bin', 'ps_calc_20_bin']

#bin_cols=['ps_ind_06_bin', 'ps_ind_07_bin','ps_ind_08_bin', 'ps_ind_09_bin', 'ps_ind_10_bin', 

         # 'ps_ind_11_bin','ps_ind_12_bin', 'ps_ind_13_bin','ps_ind_16_bin', 'ps_ind_17_bin', 

          #'ps_ind_18_bin','ps_calc_15_bin', 'ps_calc_16_bin', 'ps_calc_17_bin', 'ps_calc_18_bin',

       #'ps_calc_19_bin', 'ps_calc_20_bin']

numeric_cols=['ps_ind_01','ps_ind_03','ps_ind_14', 'ps_ind_15','ps_reg_01',

       'ps_reg_02', 'ps_reg_03','ps_car_11', 'ps_car_12', 'ps_car_13', 'ps_car_14',

       'ps_car_15', 'ps_calc_01', 'ps_calc_02', 'ps_calc_03', 'ps_calc_04',

       'ps_calc_05', 'ps_calc_06', 'ps_calc_07', 'ps_calc_08', 'ps_calc_09',

       'ps_calc_10', 'ps_calc_11', 'ps_calc_12', 'ps_calc_13', 'ps_calc_14']
#data_cat=train[cat_cols]

#test_cat=test[cat_cols]

#print(test_cat.shape)

#data_bin=train[bin_cols]

#print(data_bin.shape)

#data_num=train[numeric_cols]

#test_num=test[numeric_cols]

#print(test_num.shape)
#data_cat=data_cat.astype('object')

#test_cat=test_cat.astype('object')

#data_bin=data_bin.astype('object')
#train_data=pd.concat([data_num,data_cat],1)

#test_data=pd.concat([test_num,test_cat],1)

#test_data.shape
train_x = train.drop(['target', 'id'], axis=1)

test_x = test.drop(['id'], axis=1)

X_train, X_validation, y_train, y_validation = train_test_split(train_x, Y, train_size=0.9, random_state=1234)
dtrain = xgb.DMatrix(X_train, y_train)

dvalidation = xgb.DMatrix(X_validation, y_validation)

dtest = xgb.DMatrix(test_x)
param = {}

param['objective'] = 'binary:logistic'

param['eta'] = 0.02

param['silent'] = True

param['max_depth'] = 5

param['subsample'] = 0.8

param['colsample_bytree'] = 0.8

param['eval_metric'] = 'auc'
evallist  = [(dvalidation,'eval'), (dtrain,'train')]
model=xgb.train(param, dtrain, 963, evallist, early_stopping_rounds=100, maximize=True, verbose_eval=9)
pred = model.predict(dtest)

#y_validation=list(y_validation)
#pred=np.where(pred>0.5,1,0)
#f1_score(y_validation, pred, average='weighted')
#confusion_matrix(y_validation, pred)
sub = pd.DataFrame()

sub['id'] =test.id

sub['target'] = pred

sub.to_csv('xgboost.csv', index=False,float_format='%.2f')



print(sub.head())