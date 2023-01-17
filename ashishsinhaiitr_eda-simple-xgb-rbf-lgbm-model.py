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
train=pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')
train.head(3)
train.isnull().sum()
test.isnull().sum()
train.shape
test.shape
train.dtypes
train.describe()
test.describe()
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(20,10))
sns.distplot(train['target'],kde=True)
plt.figure(figsize=(20,10))
sns.distplot(np.log(1+train['target']))
plt.show()
from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
def rmsle(y,pred):
    return np.sqrt(np.mean(np.power(np.log1p(y)-np.log1p(pred), 2)))
from sklearn.model_selection import train_test_split
train['log_target']=np.log(1+train['target'])
train['log_target'].describe()
y=train['log_target']
train.drop(['target','log_target'], axis=1, inplace=True)
test_id=test['ID']
train_id=train['ID']
test.drop(['ID'], axis=1, inplace=True)
train.drop(['ID'], axis=1, inplace=True)

X_train,X_test,y_train,y_test=train_test_split(train,y,test_size=0.2,random_state=42)
X_train.shape,X_test.shape
from sklearn.cross_validation import StratifiedKFold,KFold
from sklearn.linear_model import ElasticNet,LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

import lightgbm as lgb
import time
lgbm_params =  {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'rmse',
    'max_depth': 8,
    'num_leaves': 32,  # 63, 127, 255
    'feature_fraction': 0.8, # 0.1, 0.01
    'bagging_fraction': 0.8,
    'learning_rate': 0.01, #0.00625,#125,#0.025,#05,
    'verbose': 1
}
model1=RandomForestRegressor(n_jobs=-1,n_estimators=42)
model1.fit(ss.fit_transform(X_train),y_train)
print(rmsle(y_test, model1.predict(ss.transform(X_test))))
pred1=model1.predict(test)
pred1
sub = pd.read_csv('../input/sample_submission.csv')
sub['target'] = pd.Series(pred1)
sub.to_csv('sub_rbf_baseline.csv', index=False)
Y_target = []
for fold_id,(train_idx, val_idx) in enumerate(KFold(n=train.shape[0],n_folds=10, random_state=42,shuffle=True)):
    print('FOLD:',fold_id)
    X_train = train.values[train_idx]
    y_train = y.values[train_idx]
    X_valid = train.values[val_idx]
    y_valid =  y.values[val_idx]
    
    
    lgtrain = lgb.Dataset(X_train, y_train,
                feature_name=train.columns.tolist(),
    #             categorical_feature = categorical
                         )

    lgvalid = lgb.Dataset(X_valid, y_valid,
                feature_name=train.columns.tolist(),
    #             categorical_feature = categorical
                         )

    modelstart = time.time()
    lgb_clf = lgb.train(
        lgbm_params,
        lgtrain,
        num_boost_round=30000,
        valid_sets=[lgtrain, lgvalid],
        valid_names=['train','valid'],
        early_stopping_rounds=100,
        verbose_eval=100
    )
    
    test_pred = lgb_clf.predict(test.values)
    Y_target.append(np.exp(test_pred)-1)
    print('fold finish after', time.time()-modelstart)
Y_target = np.array(Y_target)
Y_target.shape
sub = pd.read_csv('../input/sample_submission.csv')
sub['target'] = Y_target.mean(axis=0)
sub.to_csv('sub_lgb_baseline.csv', index=False)
