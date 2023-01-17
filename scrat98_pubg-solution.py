# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import itertools

train_file = "../input/train_V2.csv"
test_file = "../input/test_V2.csv"
sample_submission_file = "./data/sample_submission_V2.csv"

# Memory saving function credit to https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
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
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df


def read_data_with_reduce(file_name):
    print('Reading ' + file_name)
    data = pd.read_csv(file_name)
    print('Reducing ' + file_name)
    data = reduce_mem_usage(data)
    return data

train = read_data_with_reduce(train_file)
test = read_data_with_reduce(test_file)
train.info()
null_cnt = train.isnull().sum().sort_values()
print(null_cnt[null_cnt > 0])
display(train[train.isnull().any(1)])
display(test[test.isnull().any(1)])

train.dropna(inplace=True)
# train.drop(2744604, inplace=True)
train.describe(include=np.number).drop('count').T
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.model_selection import cross_val_score
import sklearn.metrics
from sklearn.model_selection import GridSearchCV

def initial_data_processing(df):
    # remove complexity fields without data analysis
    df.drop(columns=['killPoints','rankPoints','winPoints','matchType','maxPlace','Id'],inplace=True)
    X = df.groupby(['matchId','groupId']).agg(np.mean)
    y = X['winPlacePerc']
    X.drop(columns=['winPlacePerc'],inplace=True)
    X_ranked = X.groupby('matchId').rank(pct=True)
    X = X.reset_index()[['matchId','groupId']].merge(X_ranked, how='left', on=['matchId', 'groupId'])
    X.drop(['matchId','groupId'],axis=1, inplace=True)
    return X, y

X_train, y = initial_data_processing(train)
X_train, X_holdout, y_train, y_holdout = train_test_split(X_train, y, test_size=0.2)
lgtrain = lgb.Dataset(X_train, label=y_train.reset_index(drop=True))
res = lgb.cv({'metric': 'mae'}, lgtrain, nfold=5)
print("Mean score:",res['l1-mean'][-1])
gc.collect()
# TODO
gridParams = {
    'num_leaves': [30,50,100],
    'max_depth': [-1,8,15], 
    'min_data_in_leaf': [100,300,500],
    'max_bin': [250,500], 
    'lambda_l1': [0.01],
    'num_iterations': [5], 
    'nthread': [4],
    'learning_rate': [0.05],
    'metric': ['mae'],
    "bagging_fraction" : [0.7],
    "bagging_seed" : [0],
    "colsample_bytree" : [0.7]
    }
model = lgb.LGBMRegressor()
grid = GridSearchCV(model,
                    gridParams,
                    verbose=1,
                    cv=5)
grid.fit(X_train.iloc[:500000,:], y_train.iloc[:500000])
print("Best params:", grid.best_params_, '\n')
print("Best score:", grid.best_score_)
params = grid.best_params_
lgtrain = lgb.Dataset(X_train, label=y_train)
lgval = lgb.Dataset(X_holdout, label=y_holdout)
model = lgb.train(params, lgtrain, valid_sets=[lgtrain, lgval], early_stopping_rounds=200, verbose_eval=1000)
pred_test = model.predict(X_test, num_iteration=model.best_iteration)

# ids_after['winPlacePerc'] = pred_test
# predict = ids_init.merge(ids_after, how='left', on=['groupId',"matchId"])['winPlacePerc']
df_sub = pd.read_csv(sample_submission_file)
df_sub['winPlacePerc'] = pred_test
df_sub[["Id", "winPlacePerc"]].to_csv("submission.csv", index=False)