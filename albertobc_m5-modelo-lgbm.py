%reset -f

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
from fbprophet import Prophet
from itertools import cycle
import warnings
warnings.filterwarnings('ignore')
import dask.dataframe as dd
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
import lightgbm as lgb
import dask_xgboost as xgb
import dask.dataframe as dd
from sklearn import preprocessing, metrics
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold, GroupKFold, GridSearchCV, train_test_split, TimeSeriesSplit
import gc
import os
from  datetime import datetime, timedelta
import gc
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor #
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import eli5
from eli5.sklearn import PermutationImportance

def label(X, categorical_cols):

    for col in categorical_cols:
                 
        le = LabelEncoder()
        #not_null = df[col][df[col].notnull()]
        X[col] = X[col].fillna('nan')
        X[col] = pd.Series(le.fit_transform(X[col]), index=X.index)

    return X

def imputation(data):
    
    # Imputation
    my_imputer = SimpleImputer()
    imputed_data = pd.DataFrame(my_imputer.fit_transform(data))

    # Imputation removed column names; put them back
    imputed_data.columns = data.columns
    
    return imputed_data

params = {'min_child_weight': 0.034,
          'feature_fraction': 0.379,
          'bagging_fraction': 0.418,
          'objective': 'regression',
          'max_depth': -1,
          'learning_rate': 0.005,
          "boosting_type": "gbdt",
          "bagging_seed": 11,
          "metric": 'rmse',
          "verbosity": -1,
          'reg_alpha': 0.3899,
          'reg_lambda': 0.648,
          'random_state': 222,
          'num_iterations' : 1200,
          'num_leaves': 128,
          "min_data_in_leaf": 100,
         }
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns: #columns毎に処理
        col_type = df[col].dtypes
        if col_type in numerics: #numericsのデータ型の範囲内のときに処理を実行. データの最大最小値を元にデータ型を効率的なものに変更
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

ss = pd.read_csv('../input/m5-forecasting-accuracy/sales_train_validation.csv')
sales = reduce_mem_usage(ss)

cr = pd.read_csv('../input/m5-forecasting-accuracy/calendar.csv')
calendar = reduce_mem_usage(cr)

ps = pd.read_csv('../input/m5-forecasting-accuracy/sell_prices.csv')
prices = reduce_mem_usage(ps)

se = pd.read_csv('../input/m5-forecasting-accuracy/sample_submission.csv')
sample = reduce_mem_usage(se)

# make a copy of the sample submission
sub = sample.copy()
# select only the rows with an id with the validation tag
sub.columns = ['id'] + ['d_' + str(1914+x) for x in range(28)]
sub = sub.loc[sub.id.str.contains('validation')]
sub = sub.melt('id', var_name='d', value_name='demand')
firstDay = 913
lastDay = 1913

numCols = [f"d_{day}" for day in range(firstDay, lastDay+1)]
catCols = ['id', 'item_id', 'store_id']

data = sales.loc[:, catCols + numCols]
data = pd.melt(data,
             id_vars = catCols,
             value_vars = [col for col in data.columns if col.startswith("d_")],
             var_name = "d",
             value_name = "demand")

data = data.merge(calendar)
features_2 = catCols + ['weekday', 'wm_yr_wk', 'event_type_1', 'event_type_2', 'demand']
data_ = data[features_2]
valid_rows = len(sub)

categorical_cols = ['weekday', 'event_type_1', 'event_type_2'] + catCols

train = data_[:-valid_rows * 2]
valid = data_[-valid_rows * 2:-valid_rows]
test = data_[-valid_rows:]

train = label(train, categorical_cols)
valid = label(valid, categorical_cols)
test = label(test, categorical_cols)

train = imputation(train)
valid = imputation(valid)
test = imputation(test)
dtrain = lgb.Dataset(train[features_2], label=train['demand'])
dvalid = lgb.Dataset(valid[features_2], label=valid['demand'])
dtest = lgb.Dataset(test[features_2], label=test['demand'])

num_round = 1000
bst = lgb.train(params, dtrain, num_round, valid_sets=[dvalid], early_stopping_rounds=10, verbose_eval=100)
ypred = bst.predict(test[features_2])
ypred = bst.predict(test[features_2])
sub['demand'] = ypred
temp = sub.pivot(index='id', columns='d', values='demand')
temp.reset_index(inplace=True)
sub['demand'] = ypred
temp = sub.pivot(index='id', columns='d', values='demand')
temp.reset_index(inplace=True)

submission = sample[['id']].copy()
submission = submission.merge(temp)
submission = pd.concat([submission, submission], axis=0)
submission['id'] = sample.id.values
submission.columns = ['id'] + ['F' + str(i) for i in range(1,29)]
submission.head()
submission.to_csv('submission_LGBM4.csv', index=False)
data = data.merge(calendar, on = "d", copy = False)
data = data.merge(prices, on = ["store_id", "item_id", "wm_yr_wk"], copy = False)
data = reduce_mem_usage(data)
cat_features = ["id", "d", 'item_id', 'dept_id','store_id', 'cat_id', 'state_id',
                "event_name_1", "event_name_2", "event_type_1", "event_type_2"]
unusedCols = ["date", "wm_yr_wk", "weekday"]
trainCols = data.columns[~data.columns.isin(unusedCols)]

data_ = data[trainCols]

data_raw = label(data_, cat_features)

valid_rows = len(sub)

train = data_raw[:-valid_rows * 2]
train = reduce_mem_usage(train)
valid = data_raw[-valid_rows * 2:-valid_rows]
valid = reduce_mem_usage(valid)
test = data_raw[-valid_rows:]
test  = reduce_mem_usage(test)
dtrain = lgb.Dataset(train[trainCols], label=train['sales'])
dvalid = lgb.Dataset(valid[trainCols], label=valid['sales'])
dtest = lgb.Dataset(test[trainCols], label=test['sales'])

num_round = 1000
bst = lgb.train(params, dtrain, num_round, valid_sets=[dvalid], early_stopping_rounds=10, verbose_eval=100)
ypred = bst.predict(test[trainCols])
ypred = bst.predict(test[trainCols])
sub['demand'] = ypred
temp = sub.pivot(index='id', columns='d', values='demand')
temp.reset_index(inplace=True)