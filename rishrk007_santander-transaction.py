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
train=pd.read_csv("../input/train.csv")

test=pd.read_csv("../input/test.csv")
train.head(5)
train.pop("ID_code")

test.pop("ID_code")
train["target"].value_counts()
y=train["target"]
train.pop("target")
#from sklearn.preprocessing import StandardScaler
#sc=StandardScaler()
#train=sc.fit_transform(train)
#test=sc.fit_transform(test)
train.describe()
from sklearn.model_selection import StratifiedKFold,KFold

import lightgbm as lgb
train.shape
test.shape
n_fold = 5

folds = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=42)
params = {'num_leaves': 8,

         'min_data_in_leaf': 80,

          'min_sum_hessian_in_leaf': 10.0,

         'objective': 'binary',

         'max_depth': 16,

        'num_leaves': 13,  

         'learning_rate': 0.0085,

         'boosting': 'gbdt',

         'bagging_freq': 5,

         'bagging_fraction': 0.38,

         'feature_fraction': 0.04,

         'bagging_seed': 11,

         'reg_alpha':  0.1302650970728192,

         'reg_lambda': 0.3603427518866501,

         'random_state': 42,

         'metric': 'auc',

         'verbosity': -1,

         'subsample': 0.81,

         'min_gain_to_split': 0.01077313523861969,

         'min_child_weight': 19.428902804238373,

         'num_threads': 8}
prediction = np.zeros(len(test))

for fold_n, (train_index, valid_index) in enumerate(folds.split(train,y)):

    print('Fold', fold_n)

    X_train, X_valid = train.iloc[train_index], train.iloc[valid_index]

    y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

    

    train_data = lgb.Dataset(X_train, label=y_train)

    valid_data = lgb.Dataset(X_valid, label=y_valid)

        

    model = lgb.train(params,train_data,num_boost_round=20000,

                    valid_sets = [train_data, valid_data],verbose_eval=300,early_stopping_rounds = 200)

            

    #y_pred_valid = model.predict(X_valid)

    prediction += model.predict(test, num_iteration=model.best_iteration)/5
max(prediction)
sub1=pd.read_csv("../input/sample_submission.csv")
sub1["target"]=prediction
sub1.to_csv("submissionlgb.csv",index=False)
prediction