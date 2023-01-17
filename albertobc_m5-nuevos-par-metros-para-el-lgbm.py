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

def rolling_demande_fe(data):
    
    # rolling demand features
    data['lag_t28'] = data.groupby(['id'])['demand'].transform(lambda x: x.shift(28))
    data['lag_t29'] = data.groupby(['id'])['demand'].transform(lambda x: x.shift(29))
    data['lag_t30'] = data.groupby(['id'])['demand'].transform(lambda x: x.shift(30))
    data['rolling_mean_t7'] = data.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(7).mean())
    data['rolling_std_t7'] = data.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(7).std())
    data['rolling_mean_t30'] = data.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(30).mean())
    data['rolling_mean_t90'] = data.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(90).mean())
    data['rolling_mean_t180'] = data.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(180).mean())
    data['rolling_std_t30'] = data.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(30).std())
    data['rolling_skew_t30'] = data.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(30).skew())
    data['rolling_kurt_t30'] = data.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(30).kurt())
    
    return data
    
def price_fe(data):
    
    # price features
    data['lag_price_t1'] = data.groupby(['id'])['sell_price'].transform(lambda x: x.shift(1))
    data['price_change_t1'] = (data['lag_price_t1'] - data['sell_price']) / (data['lag_price_t1'])
    data['rolling_price_max_t365'] = data.groupby(['id'])['sell_price'].transform(lambda x: x.shift(1).rolling(365).max())
    data['price_change_t365'] = (data['rolling_price_max_t365'] - data['sell_price']) / (data['rolling_price_max_t365'])
    data['rolling_price_std_t7'] = data.groupby(['id'])['sell_price'].transform(lambda x: x.rolling(7).std())
    data['rolling_price_std_t30'] = data.groupby(['id'])['sell_price'].transform(lambda x: x.rolling(30).std())
    data.drop(['rolling_price_max_t365', 'lag_price_t1'], inplace = True, axis = 1)
    
    return data

def time_fe(data):
    
    # time features
    data['date'] = pd.to_datetime(data['date'])
    data['year'] = data['date'].dt.year
    data['month'] = data['date'].dt.month
    data['week'] = data['date'].dt.week
    data['day'] = data['date'].dt.day
    data['dayofweek'] = data['date'].dt.dayofweek
    
    
    return data

def add_date_features(df):
    # date time features
    df['date'] = pd.to_datetime(df['date'])
    attrs = ["year", "quarter", "month", "week", "day", "dayofweek", "is_year_end", "is_year_start", "is_quarter_end", \
        "is_quarter_start", "is_month_end","is_month_start",
    ]

    for attr in attrs:
        dtype = np.int16 if attr == "year" else np.int8
        df[attr] = getattr(df['date'].dt, attr).astype(dtype)
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(np.int8)
    return df

def create_lag_features(df):
    
    # shift and rolling demand features
    shifts = [1, 7, 28]
    for shift_window in shifts:
        df[f"shift_t{shift_window}"] = df.groupby(["id"])["demand"].transform(lambda x: x.shift(shift_window))
            
    # rolling mean
    windows = [7, 28]
    for val in windows:
        for col in [f'shift_t{win}' for win in windows]:
            df[f'roll_mean_{col}_{val}'] = df[['id', col]].groupby(['id'])[col].transform(lambda x: x.rolling(val).mean())
    
    # rolling standard deviation    
    for val in windows:
        for col in [f'shift_t{win}' for win in windows]:
            df[f'roll_std_{col}_{val}'] = df[['id', col]].groupby(['id'])[col].transform(lambda x: x.rolling(val).std())
    
    return df
ss = pd.read_csv('../input/m5-forecasting-accuracy/sales_train_validation.csv')
sales = reduce_mem_usage(ss)

cr = pd.read_csv('../input/m5-forecasting-accuracy/calendar.csv')
calendar = reduce_mem_usage(cr)
calendar.drop(['weekday', 'wday', 'month', 'year'], inplace = True, axis = 1)

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
calendar = label(
           calendar, 
           ["event_name_1", "event_type_1", "event_name_2", "event_type_2"]).pipe(reduce_mem_usage)

sales = label(
        sales, ["item_id", "dept_id", "cat_id", "store_id", "state_id"],
        ).pipe(reduce_mem_usage)

prices = label(prices, ["item_id", "store_id"]).pipe(reduce_mem_usage)
catCols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
firstDay = 913
lastDay = 1913

numCols = [f"d_{day}" for day in range(firstDay, lastDay+1)]

data = sales.loc[:, catCols + numCols]

data = pd.melt(data,
             id_vars = catCols,
             value_vars = [col for col in data.columns if col.startswith("d_")],
             var_name = "d",
             value_name = "demand")

print(f'DF shape after melting is {sales.shape}')
cal_imp_cols = ['d', 'date', 'day', 'week', 'month','quarter', 'year','is_weekend', 'event_name_1', 'event_type_1', ]

calendar = add_date_features(calendar)
calendar = calendar[cal_imp_cols]
calendar = reduce_mem_usage(calendar)

data = pd.merge(data, calendar, how = 'left', on = ['d'])
data.drop(['d'], inplace = True, axis = 1)
data = reduce_mem_usage(data)
#data = data.merge(prices, on = ["store_id", "item_id"], copy = False)
#data = rolling_demande_fe(data)
#data.drop(['id'], inplace = True, axis = 1)
x_train = data[data['date'] <= '2016-03-27']
y_train = x_train['demand']
x_val = data[(data['date'] > '2016-03-27') & (data['date'] <= '2016-04-24')]
y_val = x_val['demand']
test = data[(data['date'] > '2016-04-24')]

features = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'demand', 'day', 'week', 'month', 'year',  'quarter', 
            'is_weekend', 'event_name_1', 'event_type_1'] #, 'lag_t28', 'lag_t29', 'lag_t30']
train_set = lgb.Dataset(x_train[features], y_train)
val_set = lgb.Dataset(x_val[features], y_val)

num_round = 1000
model = lgb.train(params, train_set, num_round, valid_sets=[val_set], early_stopping_rounds=10, verbose_eval=100)
val_pred = model.predict(x_val[features])

sub['demand'] = val_pred
temp = sub.pivot(index='id', columns='d', values='demand')
temp.reset_index(inplace=True)

submission = sample[['id']].copy()
submission = submission.merge(temp)
submission = pd.concat([submission, submission], axis=0)
submission['id'] = sample.id.values
submission.columns = ['id'] + ['F' + str(i) for i in range(1,29)]
submission.head()
submission.to_csv('submission_LGBM4.csv', index=False)