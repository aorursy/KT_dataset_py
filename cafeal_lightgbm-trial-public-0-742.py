# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import random
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split
import lightgbm as lgb

from matplotlib import pyplot as plt
import seaborn as sns

seed = 1234
random.seed(seed)
np.random.seed(seed)
import os
input_files = os.listdir("../input")
print(input_files)

# read each csv file as data frame variable
for filename in input_files:
    locals()[filename.rstrip('.csv')] = pd.read_csv(f'../input/{filename}')
# categoricalize
categorical_columns = ['NAME_CONTRACT_TYPE',
                       'CODE_GENDER',
                       'FLAG_OWN_CAR',
                       'FLAG_OWN_REALTY',
                       'NAME_TYPE_SUITE',
                       'NAME_INCOME_TYPE',
                       'NAME_EDUCATION_TYPE',
                       'NAME_FAMILY_STATUS',
                       'NAME_HOUSING_TYPE',
                       'OCCUPATION_TYPE',
                       'WEEKDAY_APPR_PROCESS_START',
                       'ORGANIZATION_TYPE',
                       'FONDKAPREMONT_MODE',
                       'HOUSETYPE_MODE',
                       'WALLSMATERIAL_MODE',
                       'EMERGENCYSTATE_MODE']

for column in categorical_columns:
    application_train[column] = application_train[column].astype('category')
    application_test[column] = application_test[column].astype('category')
input_columns = application_train.columns
input_columns = input_columns[input_columns != 'TARGET']
target_column = 'TARGET'

X = application_train[input_columns]
y = application_train[target_column]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed)

lgb_train = lgb.Dataset(data=X_train, label=y_train)
lgb_eval = lgb.Dataset(data=X_test, label=y_test)

params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': {'auc'},
        'learning_rate': 0.1,
        'num_leaves': 23,
        'min_data_in_leaf': 1,
        'num_iteration': 200,
        'verbose': 0
}

# train
gbm = lgb.train(params,
            lgb_train,
            num_boost_round=50,
            valid_sets=lgb_eval,
            early_stopping_rounds=10)
lgb.plot_importance(gbm, figsize=(20, 20))
import time
pred = gbm.predict(application_test[input_columns])
submission = sample_submission
submission.TARGET = pred
submission.to_csv(f"{time.strftime('%Y_%m_%d_%d_%M')}_submission.csv", index=None)
