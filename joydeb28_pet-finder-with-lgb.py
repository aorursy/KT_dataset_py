# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
import xgboost as xgb
import lightgbm as lgbm

# Any results you write to the current directory are saved as output.
train_df = pd.read_csv("../input/train/train.csv")
test_df = pd.read_csv("../input/test/test.csv")
y = train_df.AdoptionSpeed.values
train_df = train_df.drop(['Name', 'RescuerID', 'Description', 'PetID', 'AdoptionSpeed'], axis=1).values
test_df = test_df.drop(['Name', 'RescuerID', 'Description', 'PetID'], axis=1).values

cols = ["Type", "Age", "Breed1", "Breed2", "Gender", "Color1", "Color2", "Color3", "MaturitySize", 
          "FurLength", "Vaccinated", "Dewormed", "Sterilized", "Health", "Quantity", "Fee", "State", 
          "VideoAmt", "PhotoAmt"]
#model = xgb.XGBClassifier(n_estimators=500, nthread=-1, max_depth=8, learning_rate=0.015)
import lightgbm as lgb

#d_train = lgb.Dataset(train_df, label=y)
params = {}
params['learning_rate'] = 0.003
params['boosting_type'] = 'gbdt'
params['objective'] = 'binary'
#params['metric'] = 'binary_logloss'
params['sub_feature'] = 0.5
params['num_leaves'] = 10
params['min_data'] = 50
params['max_depth'] = 10

#model = lgb.LGBMClassifier(params)
model = lgb.LGBMClassifier(boosting_type='gbdt', class_weight=None,
        colsample_bytree=0.7350947516014185, importance_type='split',
        learning_rate=0.2742463490407598, max_depth=8,
        min_child_samples=20, min_child_weight=0.03867970435532146,
        min_split_gain=0.437061075871395, n_estimators=132, n_jobs=-1,
        num_leaves=33, objective='multiclass', random_state=13579,
        reg_alpha=0.0, reg_lambda=33.454282508690014, silent=True,
        subsample=0.9374889029367177, subsample_for_bin=400000,
        subsample_freq=0)

#model.fit(train_df, y)
model.fit(train_df, y)
pred = model.predict(test_df)
sub = pd.read_csv('../input/test/sample_submission.csv')
sub.AdoptionSpeed = pred
sub.to_csv('submission.csv', index=False)