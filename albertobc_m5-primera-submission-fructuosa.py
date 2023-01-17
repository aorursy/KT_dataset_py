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
data = sales.iloc[:, pd.np.r_[0,-100:0]]
data = data.melt('id', var_name='d', value_name='demand')

data = data.merge(calendar)
media = data.groupby(['id','wday'])['demand'].mean()

# make a copy of the sample submission
sub = sample.copy()
# select only the rows with an id with the validation tag
sub.columns = ['id'] + ['d_' + str(1914+x) for x in range(28)]
sub = sub.loc[sub.id.str.contains('validation')]
sub = sub.melt('id', var_name='d', value_name='demand')

val_rows = len(sub)
def label(X, categorical_cols):

    for col in categorical_cols:
                 
        le = LabelEncoder()
        #not_null = df[col][df[col].notnull()]
        X[col] = X[col].fillna('nan')
        X[col] = pd.Series(le.fit_transform(X[col]), index=X.index)

    return X
y = data.demand
features = ['id', 'd', 'wday']
X_raw = data[features]

X = label(X_raw, ['id', 'd'])

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)


rf_model = RandomForestRegressor(random_state=1)

rf_model.fit(train_X,train_y)
days_wanted = 28
features = ['id', 'd', 'wday']
X_raw = data[features]

X = label(X_raw, ['id', 'd'])

train = X[:-days_wanted*2]
test  = X[-days_wanted*2:-days_wanted]
valid = X[-days_wanted:]
val_X = X[-val_rows:]

preds_val = rf_model.predict(val_X)
sub['demand'] = preds_val
temp = sub.pivot(index='id', columns='d', values='demand')
temp.reset_index(inplace=True)
submission = sample[['id']].copy()
submission = submission.merge(temp)
submission = pd.concat([submission, submission], axis=0)
submission['id'] = sample.id.values
submission.columns = ['id'] + ['F' + str(i) for i in range(1,29)]
submission.head()
submission.to_csv('submission.csv', index=False)