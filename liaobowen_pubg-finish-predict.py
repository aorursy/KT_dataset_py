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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
submission = pd.DataFrame(columns=['Id','winPlacePerc'])
submission.Id = df_test.Id
df_train.head()
df_train.shape
df_train.info()
df_train.describe()
df_train.drop('Id',axis=1,inplace=True)
df_test.drop('Id',axis=1,inplace=True)
# for col in df_train.columns:
#     sns.distplot(df_train[col])
#     plt.show()
# df_train_corr = df_train.corr().abs()
# plt.figure(figsize=(20,15))
# sns.heatmap(df_train_corr,annot=True)
# plt.show()
df_train.drop(['numGroups','killPlace', 'roadKills','swimDistance', 'teamKills', 'vehicleDestroys'],axis=1,inplace=True)
df_test.drop(['numGroups','killPlace', 'roadKills','swimDistance', 'teamKills', 'vehicleDestroys'],axis=1,inplace=True)
# df_train.drop(['numGroups','groupId','matchId'],axis=1,inplace=True)
# df_test.drop(['numGroups','groupId','matchId'],axis=1,inplace=True)

df_train_mean = df_train.groupby(['matchId','groupId']).mean().reset_index()
df_test_mean = df_test.groupby(['matchId','groupId']).mean().reset_index()

df_train_min = df_train.groupby(['matchId','groupId']).min().reset_index()
df_test_min = df_test.groupby(['matchId','groupId']).mean().reset_index()

df_train_max = df_train.groupby(['matchId','groupId']).max().reset_index()
df_test_max = df_test.groupby(['matchId','groupId']).max().reset_index()

df_train = pd.merge(df_train,df_train_mean,suffixes=['','_mean'],how='left',on=['matchId','groupId'])
df_test = pd.merge(df_test,df_test_mean,suffixes=['','_mean'],how='left',on=['matchId','groupId'])
df_train = pd.merge(df_train,df_train_min,suffixes=['','_min'],how='left',on=['matchId','groupId'])
df_test = pd.merge(df_test,df_test_min,suffixes=['','_min'],how='left',on=['matchId','groupId'])
df_train = pd.merge(df_train,df_train_max,suffixes=['','_max'],how='left',on=['matchId','groupId'])
df_test = pd.merge(df_test,df_test_max,suffixes=['','_max'],how='left',on=['matchId','groupId'])

col_list = []
for col in df_test.columns:
    if '_' in col:
        col_list.append(col)
        
y = df_train.winPlacePerc
X = df_train[col_list]

df_test = df_test[col_list]

del df_train,df_train_mean,df_test_mean,df_test_min,df_train_min,df_train_max,df_test_max
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error as MAE
x_train,x_test,y_train,y_test = train_test_split(X,y,random_state=64,test_size=0.2)

# forest = RandomForestRegressor()
# forest.fit(x_train,y_train)
# y_predict_ = forest.transform(x_train)
# y_predict = forest.transform(s_test)
# print('forest_MAE in train : {}'.format(MAE(y_train,y_predict_)))
# print('forest_MAE in test :{}'.format(MAE(y_test,y_predict)))

from sklearn.feature_selection import SelectKBest,f_classif

# selector = SelectKBest(f_classif,k=10)
# selector.fit(x_train,y_train)
# score_p = selector.pvalues_
# score_s = selector.scores_
# plt.figure(figsize=(18,9))
# plt.subplot(211)
# plt.bar(range(len(score_p)),-np.log(score_p))
# plt.xticks(range(len(score_p)),X.columns,rotation=45)
# plt.title('')
# plt.subplot(212)
# plt.bar(range(len(score_s)),score_s)
# plt.xticks(range(len(score_s)),X.columns,rotation=45)
# plt.show()
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X,y)
y_predict_ = lr.predict(x_train)
y_predict = lr.predict(x_test)

print('forest_MAE in train : {}'.format(MAE(y_train,y_predict_)))
print('forest_MAE in test :{}'.format(MAE(y_test,y_predict)))
from sklearn.linear_model import LassoCV
from lightgbm import LGBMRegressor

lgb1 = LGBMRegressor(max_depth=6,learning_rate=0.2)
lgb1.fit(X,y)
y_predict_ = lgb1.predict(x_train)
y_predict = lgb1.predict(x_test)

print("MAE in train: {}".format(MAE(y_train,y_predict_)))
print('MAE in test : {}'.format(MAE(y_test,y_predict)))
submission.winPlacePerc = lgb1.predict(df_test)
submission.to_csv('samble_submission.csv',index=False)
# from xgboost import XGBRegressor

# xgb1 = XGBRegressor(max_depth=8,learning_rate=0.05)
# xgb1.fit(x_train,y_train)
# y_predict_ = xgb1.predict(x_train)
# y_predict = xgb1.predict(x_test)
# print('MAE in train :{}'.format(MAE(y_train,y_predict_)))
# print('MAE in test  :{}'.format(MAE(y_test,y_predict)))


params = {
          'boosting_type': 'gbdt', 
          'objective':'regression',
          'silent': 0,
          'learning_rate': 0.1, 
          'max_depth': 6,
          'max_bin': 127, 
          'subsample_for_bin': 50000,
          'subsample': 0.8, 
          'colsample_bytree': 0.8, 
          'min_child_weight': 1, 
}
import lightgbm as lgb
def modelfit(params,model,x_train,y_train,early_stopping_rounds=10):
    
    lgb_params = params.copy()
    
    lgb_train = lgb.Dataset(x_train,y_train,silent=False)
    
    cv_result = lgb.cv(
        lgb_params,
        lgb_train,
        num_boost_round=10000,
        nfold=10,
        stratified=False,
        shuffle=True,
        metrics='mean_absolute_error',
        early_stopping_rounds=early_stopping_rounds,
    )
    cv.to_csv('cv_result.csv')
    
# modelfit(params,lgb1,x_train,y_train)
