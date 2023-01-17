# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

sns.set_style('whitegrid')

import time

import lightgbm as lgb

from sklearn.model_selection import KFold,StratifiedKFold

from sklearn.model_selection import GridSearchCV

import xgboost as xgb

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv('../input/train.csv')

print('Rows: ',train_df.shape[0],'Columns: ',train_df.shape[1])

train_df.info()
train_df.head()
train_df['target'].value_counts()
sns.countplot(train_df['target'])

sns.set_style('whitegrid')
test_df = pd.read_csv('../input/test.csv')
X_test = test_df.drop('ID_code',axis=1)
X = train_df.drop(['ID_code','target'],axis=1)

y = train_df['target']
n_fold = 5

folds = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=42)
params = {'num_leaves': 8,

         'min_data_in_leaf': 42,

         'objective': 'binary',

         'max_depth': 16,

         'learning_rate': 0.0123,

         'boosting': 'gbdt',

         'bagging_freq': 5,

         'bagging_fraction': 0.8,

         'feature_fraction': 0.8201,

         'bagging_seed': 11,

         'reg_alpha': 1.728910519108444,

         'reg_lambda': 4.9847051755586085,

         'random_state': 42,

         'metric': 'auc',

         'verbosity': -1,

         'subsample': 0.81,

         'min_gain_to_split': 0.01077313523861969,

         'min_child_weight': 19.428902804238373,

         'num_threads': 4}
prediction = np.zeros(len(X_test))

for fold_n, (train_index, valid_index) in enumerate(folds.split(X,y)):

    print('Fold', fold_n, 'started at', time.ctime())

    X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]

    y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

    

    train_data = lgb.Dataset(X_train, label=y_train)

    valid_data = lgb.Dataset(X_valid, label=y_valid)

        

    model = lgb.train(params,train_data,num_boost_round=20000,

                    valid_sets = [train_data, valid_data],verbose_eval=300,early_stopping_rounds = 200)

            

    #y_pred_valid = model.predict(X_valid)

    prediction += model.predict(X_test, num_iteration=model.best_iteration)/5
from catboost import CatBoostClassifier,Pool

prediction1 = np.zeros(len(X_test))

m = CatBoostClassifier(loss_function="Logloss",eval_metric="AUC",

                       boosting_type = 'Ordered')

for fold_n, (train_index, valid_index) in enumerate(folds.split(X,y)):

    print('Fold', fold_n, 'started at', time.ctime())

    X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]

    y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]



    train_data = Pool(X_train, label=y_train)

    valid_data = Pool(X_valid, label=y_valid)



    model1 = m.fit(train_data,eval_set=valid_data,use_best_model=True,verbose=300)

    

    prediction1 += model1.predict(X_test)/5
mod = xgb.XGBClassifier(max_depth=4,n_estimators=999999, colsample_bytree=0.7,subsample = 0.7, 

                              min_child_weight = 50, eval_metric = "auc",gamma = 5,alpha = 0,

                               booster = "gbtree",colsample_bylevel = 0.7, learning_rate=0.1,

                              objective='binary:logistic', n_jobs=-1)



prediction2 = np.zeros(len(X_test))

for fold_n, (train_index, valid_index) in enumerate(folds.split(X,y)):

    print('Fold', fold_n, 'started at', time.ctime())

    X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]

    y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

    

    #evallist = [(valid_data, 'eval'), (train_data, 'train')]

    model2 = mod.fit(X_train, y_train, eval_set=[(X_valid, y_valid)],verbose=200, eval_metric='auc',

                        early_stopping_rounds=200)

    

    prediction2 += model2.predict(X_test, ntree_limit=model2.best_ntree_limit)/5
prediction
sub = pd.DataFrame({"ID_code": test_df.ID_code.values})

sub["target"] = prediction

sub.to_csv("submission.csv", index=False)
sub["target"] = prediction1

sub.to_csv("submission1.csv", index=False)
sub["target"] = (prediction + prediction1)/2

sub.to_csv("submission2.csv", index=False)
sub["target"] = prediction2

sub.to_csv("submission3.csv", index=False)
sub["target"] = (prediction + prediction2)/2 

sub.to_csv("submission4.csv", index=False)
sub["target"] = (prediction + prediction1 + prediction2)/3 

sub.to_csv("submission5.csv", index=False)