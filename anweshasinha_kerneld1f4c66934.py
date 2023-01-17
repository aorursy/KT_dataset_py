# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

import pandas as pd
import numpy as np
train= pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from collections import Counter
#remove the ID field from the train data
train = train.drop('ID', axis = 1)
test = test.drop('ID', axis=1)
train.columns

#no categorical variables.
train.select_dtypes(include=['object']).dtypes

test.shape
train.shape
train['target'].describe()
x_train = train.iloc[:,train.columns!='target']
y_train = train.iloc[:,train.columns=='target']
x_test = test
drop_cols=[]
for cols in x_train.columns:
    if x_train[cols].std()==0:
        drop_cols.append(cols)
print("Number of constant columns to be dropped: ", len(drop_cols))
print(drop_cols)
x_train.drop(drop_cols,axis=1, inplace = True)
drop_cols_test=[]
for cols in x_test.columns:
    if x_test[cols].std()==0:
        drop_cols_test.append(cols)
print("Number of constant columns to be dropped: ", len(drop_cols_test))
print(drop_cols_test)
x_test.drop(drop_cols,axis=1, inplace = True)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
print(x_train)
from sklearn.decomposition import PCA
pca_x = PCA(0.95).fit(x_train)
print('%d components explain 95%% of the variation in data' % pca_x.n_components_)
pca = PCA(n_components=1600)
#fit
pca.fit(x_train)
#transform on train data
x_train_pca = pca.transform(x_train)
#transform on test data
x_test_pca = pca.transform(x_test)
import xgboost as xgb
import lightgbm as lgb
model_xgb = xgb.XGBRegressor(colsample_bytree=0.055, colsample_bylevel =0.5, 
                             gamma=1.5, learning_rate=0.02, max_depth=32, 
                             objective='reg:linear',booster='gbtree',
                             min_child_weight=57, n_estimators=1000, reg_alpha=0, 
                             reg_lambda = 0,eval_metric = 'rmse', subsample=0.7, 
                             silent=1, n_jobs = -1, early_stopping_rounds = 14,
                             random_state =7, nthread = -1)
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=144,
                              learning_rate=0.005, n_estimators=720, max_depth=13,
                              metric='rmse',is_training_metric=True,
                              max_bin = 55, bagging_fraction = 0.8,verbose=-1,
                              bagging_freq = 5, feature_fraction = 0.9) 
model_xgb.fit(x_train_pca, y_train)
model_lgb.fit(x_train_pca, y_train)
pred_xgb = model_xgb.predict(x_test_pca)
pred_lgb = model_lgb.predict(x_test_pca)
pred_xgb
pred_lgb
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
rf.fit(x_train_pca, y_train)
rf_pca_predict = rf.predict(x_test_pca)
rf_pca_predict
print(len(rf_pca_predict))
submission = pd.read_csv('../input/sample_submission.csv')
submission["target"] = pred_xgb
submission.shape
print(submission.head())
submission.to_csv('amf.csv', index=False)
