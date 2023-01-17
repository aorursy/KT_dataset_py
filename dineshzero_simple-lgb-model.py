# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from random import sample

from sklearn.preprocessing import LabelEncoder



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train=pd.read_csv("../input/train.csv")
test=pd.read_csv("../input/test.csv")
for i in train.dtypes:

    if i !=float:

        print(i)
train['ID_code'].dtypes
target=train.target

del train['target']

del train['ID_code']
submission=pd.read_csv('../input/sample_submission.csv')
del test['ID_code']
from sklearn.model_selection import train_test_split
train_x,val_x,train_y,val_y=train_test_split(train,target,test_size=0.2)
import xgboost as xgb
model = xgb.XGBClassifier(max_depth=2,

                              n_estimators=999999,

                              colsample_bytree=0.3,

                              learning_rate=0.02,

                              objective='binary:logistic', 

                              n_jobs=-1)
model.fit(train_x,train_y,eval_set=[(val_x,val_y)],verbose=0,early_stopping_rounds=200)
predicted_xgb=model.predict_proba(test)
import lightgbm as lgb
lgb_train=lgb.Dataset(train_x,train_y)

lgb_val=lgb.Dataset(val_x,val_y)
lgb_params = {

        "objective" : "binary",

        "metric" : "tweedie",

        "max_depth" : 2,

        "num_leaves" : 2,

        "learning_rate" : 0.055,

        "bagging_fraction" : 0.3,

        "feature_fraction" : 0.15,

        "lambda_l1" : 5,

        "lambda_l2" : 5,

        "verbosity" : 1

    }
model = lgb.train(lgb_params, lgb_train, 30000, valid_sets=[lgb_val], early_stopping_rounds=100, verbose_eval=100)
predicted=model.predict(test)
#from sklearn.neural_network import MLPRegressor
#model1=MLPRegressor(hidden_layer_sizes=(200,200,200))

#model1.fit(train,target)

#predicted1=model1.predict(test)
predicted1=model1.predict_proba(test)
predicted_xgb=predicted_xgb.transpose()[1]
submission['target']=(predicted+predicted_xgb)/2
submission.loc[submission['target']>=0.5,'target']=1

submission.loc[submission['target']<0.5,'target']=0
prediction=pd.DataFrame()

prediction['xgb']=predicted_xgb

prediction['lgb']=predicted

prediction['mlp']=predicted1
prediction.describe()
submission.to_csv('submission.csv',index=False)
prediction.to_csv('prediction.csv',index=False)