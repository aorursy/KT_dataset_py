# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import numpy as np
train_org    = pd.read_csv('../input/new-york-taxi-fare-prediction/train_data_org.csv')
test_org     = pd.read_csv('../input/new-york-taxi-fare-prediction/test.csv')
y =            pd.read_csv('../input/new-york-taxi-fare-prediction/train_labels.csv') 
train_scaled = pd.read_csv('../input/new-york-taxi-fare-prediction/trained_scaled.csv')
test_scaled  = pd.read_csv('../input/new-york-taxi-fare-prediction/testdata_scaled.csv')
train_org.head()
import lightgbm as lgbm

params = {
        'boosting_type':'gbdt',
        'objective': 'regression',
        'nthread': -1,
        'verbose': 0,
        'num_leaves': 31,
        'learning_rate': 0.05,
        'max_depth': -1,
        'subsample': 0.8,
        'subsample_freq': 1,
        'colsample_bytree': 0.6,
        'reg_aplha': 1,
        'reg_lambda': 0.001,
        'metric': 'rmse',
        'min_split_gain': 0.5,
        'min_child_weight': 1,
        'min_child_samples': 10,
        'scale_pos_weight':1     
    }

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(train_scaled,y,random_state=123,test_size=0.10)
train_set = lgbm.Dataset(x_train, y_train, silent=False)
valid_set = lgbm.Dataset(x_test, y_test, silent=False)
model = lgbm.train(params, train_set = train_set, num_boost_round=10000,early_stopping_rounds=500,verbose_eval=500, valid_sets=valid_set)
predict = model.predict(test_scaled,num_iteration=model.best_iteration)
#prediction = model.predict(df_test, num_iteration = model.best_iteration) 
submission = pd.read_csv('../input/new-york-city-taxi-fare-prediction/sample_submission.csv')
test = pd.read_csv('../input/new-york-city-taxi-fare-prediction/test.csv')
submission = pd.DataFrame({
        "key": test["key"],
        "fare_amount": predict
      
})

submission.to_csv('taxi_fare_submission911.csv',index=False)
submission.head()
test.head()
