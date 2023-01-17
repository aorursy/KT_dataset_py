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
test = pd.read_csv("../input/loan-default-prediction/test_v2.csv.zip", low_memory=True)

train = pd.read_csv("../input/loan-default-prediction/train_v2.csv.zip", low_memory=True)

submission = pd.read_csv("../input/loan-default-prediction/sampleSubmission.csv")
def reduce_mem_usage(df, verbose=True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024**2    

    for col in df.columns:

        col_type = df[col].dtypes

        if col_type in numerics:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)    

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df
train.info()
train = reduce_mem_usage(train, verbose=True)
train.info()
non_number_train_columns = train.dtypes[train.dtypes == object].index.values

non_number_train_columns
for col in non_number_train_columns:

    print(col, len(train[train[col]=='NA']))
train.fillna(-1)



for cols in train[train['f137']=='NA'].index:

    train.loc[cols,'f137'] = -1



for cols in non_number_train_columns:

    train[cols] = train[cols].astype(np.float64)
train.info()
test.info()
test = reduce_mem_usage(test, verbose=True)
test.info()
test.head()
non_number_test_columns = test.dtypes[test.dtypes == object].index.values

non_number_test_columns
for i in test[test['f276']=='NA'].index:

    print(i)
for cols in test[test['f276']=='NA'].index:

    print(test.loc[cols,'f276'])
# test.loc[cols,'f276']
test[test['f276']=='NA']
# test.loc[test[test['f137']=='NA'],'f137']
test.loc[1,'f137']
test.fillna(-1)



for cols in test[test['f276']=='NA'].index:

    test.loc[cols,'f276'] = -1



for cols in non_number_test_columns:

    test[cols] = test[cols].astype(np.float64)
test.info()
train = train.fillna(train.mean())

test = test.fillna(test.mean())
x_train = train.iloc[:len(train)*9//10].drop(['id','loss'], axis=1)

x_val = train.iloc[len(train)*9//10:].drop(['id','loss'], axis=1)

x_test = test.drop(['id'], axis=1)



y_train = train.iloc[:len(train)*9//10]['loss']

y_val = train.iloc[len(train)*9//10:]['loss']
import time

from xgboost import XGBRegressor

ts = time.time()



model = XGBRegressor(

    max_depth=10,

    n_estimators=1000,

    min_child_weight=0.5, 

    colsample_bytree=0.8, 

    subsample=0.8, 

    eta=0.1,

#     tree_method='gpu_hist',

    seed=42)



model.fit(

    x_train, 

    y_train, 

    eval_metric="rmse", 

    eval_set=[(x_train, y_train), (x_val, y_val)], 

    verbose=True, 

    early_stopping_rounds = 20)



time.time() - ts
Y_pred = model.predict(x_val)

Y_test = model.predict(x_test)
submission['loss'] = Y_test

submission.to_csv('submission_xgb.csv',index=False)
# test.isnull().sum()