# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import lightgbm as LGB  

from sklearn.metrics import roc_auc_score, make_scorer

from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split

from itertools import product



train = pd.read_csv('/kaggle/input/cat-in-the-dat/train.csv')



train.head()
test = pd.read_csv('/kaggle/input/cat-in-the-dat/test.csv')



test.head()
train_size=train.shape[0]

test_size=test.shape[0]

print('Train size = ' + str(train_size))

print('Test size = ' + str(test_size))
y_train_tmp=train['target']



whole_dataset=pd.concat([train.drop('target',axis=1), test], axis=0)

whole_dataset = whole_dataset.drop(['id', ], axis=1)

whole_dataset.head()
whole_dataset.isnull().sum()
OneHot=OneHotEncoder(drop='first', sparse=True)

OneHot.fit(whole_dataset)

OH_train_tmp=OneHot.transform(whole_dataset.iloc[:train_size,:])

OH_test=OneHot.transform(whole_dataset.iloc[train_size:,:])

OH_train, OH_val, y_train, y_val = train_test_split(OH_train_tmp, y_train_tmp, test_size=0.05, random_state=9)



lgb_train = LGB.Dataset(OH_train, y_train)  

lgb_eval = LGB.Dataset(OH_val, y_val, reference=lgb_train) 

params = {  

    'boosting_type': 'gbdt',  

    'objective': 'binary',  

    'metric': 'auc', 

    'max_depth': 2,  

    'learning_rate': 0.3,  

    'feature_fraction': 0.2,

    'is_unbalance': True  

}  
gbm = LGB.train(params,  

          lgb_train,  

          num_boost_round=10000,  

          valid_sets=[lgb_train, lgb_eval, ],  

          early_stopping_rounds=500,

          verbose_eval=200) 
y_train_pred_lgb=gbm.predict(OH_train, num_iteration=gbm.best_iteration)

y_val_pred_lgb=gbm.predict(OH_val, num_iteration=gbm.best_iteration)

print("Training auc : ",roc_auc_score(y_train, y_train_pred_lgb))

print("Val auc : ",roc_auc_score(y_val, y_val_pred_lgb))
y_test_pred_lgb=gbm.predict(OH_test, num_iteration=gbm.best_iteration) 
y_test_pred = y_test_pred_lgb

submission=pd.DataFrame({'id': np.arange(300000, 500000,1), 'target':y_test_pred})

submission.to_csv('submission.csv', index=False)