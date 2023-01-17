import gc

import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold

import optuna
import lightgbm as lgb

from sklearn.metrics import mean_squared_error

import warnings
warnings.filterwarnings('ignore')
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
def replace_inf2nan(df):
    df = df.replace(np.inf, np.nan)
    df = df.replace(-np.inf, np.nan)
    return df
sales_train_validation = reduce_mem_usage(pd.read_csv('../input/m5-forecasting-accuracy/sales_train_validation.csv'))
sell_prices            = reduce_mem_usage(pd.read_csv('../input/m5-forecasting-accuracy/sell_prices.csv'))
calendar               = reduce_mem_usage(pd.read_csv('../input/m5-forecasting-accuracy/calendar.csv'))
sample_submission      = reduce_mem_usage(pd.read_csv('../input/m5-forecasting-accuracy/sample_submission.csv'))
idsList = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']

sales_train_validation = sales_train_validation.melt(id_vars=idsList, var_name='d', value_name='target')

test1_rows = [row for row in sample_submission['id'] if 'validation' in row]
test2_rows = [row for row in sample_submission['id'] if 'evaluation' in row]
test1 = sample_submission[sample_submission['id'].isin(test1_rows)]
test2 = sample_submission[sample_submission['id'].isin(test2_rows)]

test1.columns = ['id', 'd_1914', 'd_1915', 'd_1916', 'd_1917', 'd_1918', 'd_1919', 'd_1920', 'd_1921', 'd_1922', 'd_1923',
                 'd_1924', 'd_1925', 'd_1926', 'd_1927', 'd_1928', 'd_1929', 'd_1930', 'd_1931', 'd_1932', 'd_1933', 'd_1934', 
                 'd_1935', 'd_1936', 'd_1937', 'd_1938', 'd_1939', 'd_1940', 'd_1941']
test2.columns = ['id', 'd_1942', 'd_1943', 'd_1944', 'd_1945', 'd_1946', 'd_1947', 'd_1948', 'd_1949', 'd_1950', 'd_1951', 
                 'd_1952', 'd_1953', 'd_1954', 'd_1955', 'd_1956', 'd_1957', 'd_1958', 'd_1959', 'd_1960', 'd_1961', 'd_1962', 
                 'd_1963', 'd_1964', 'd_1965', 'd_1966', 'd_1967', 'd_1968', 'd_1969']

product = sales_train_validation[idsList].drop_duplicates()

test1 = test1.merge(product, how='left', on='id')

test2['id'] = test2['id'].str.replace('_evaluation', '_validation')
test2 = test2.merge(product, how='left', on='id')
test2['id'] = test2['id'].str.replace('_validation', '_evaluation')

test1 = pd.melt(test1, id_vars=idsList, var_name='d', value_name='target')
test2 = pd.melt(test2, id_vars=idsList, var_name='d', value_name='target')

sales_train_validation['part'] = 0
test1['part'] = 1
test2['part'] = 2

train = pd.concat([sales_train_validation, test1, test2], axis=0)

submission = train.loc[train['part'].isin([1, 2]), ['id', 'd', 'target']]

train['id_'] = train['id'].str.replace('_validation', '')
train['id_'] = train['id_'].str.replace('_evaluation', '')

del idsList, sales_train_validation, test1_rows, test2_rows, test1, test2, product
gc.collect()
calendar['day'] = calendar['date'].str[8:10]
calendar['month_day'] = calendar['month'].astype(str) + calendar['day'].astype(str)

calendar['Christmas'] = (calendar['date'].str[5:]=='12-25').astype(int)
calendar['BlackFriday'] = (calendar['date'].str[5:].isin(['2011-11-25', '2012-11-23', '2013-11-29', '2014-11-28', '2015-11-27'])).astype(int)

calendar.loc[calendar['BlackFriday']==1, 'event_name_1'] = 'BlackFriday'
calendar.loc[calendar['BlackFriday']==1, 'event_type_1'] = 'other'

calendar.fillna('NoEvent', inplace=True)

calendar['yesterday_event'] = calendar['event_name_1'].shift(1)
calendar['tommorrow_event'] = calendar['event_name_1'].shift(-1)

calendar['month_day'] = calendar['month'].astype(str) + calendar['day'].astype(str)

calendar['weekend'] = (calendar['weekday'].isin(['Saturday', 'Sunday'])).astype(int)
calendar.loc[calendar['event_type_1']=='National', 'weekend'] = 1
price_mode = pd.DataFrame(sell_prices.groupby(['item_id'])['sell_price'].apply(lambda x: x.mode()))
price_mode = price_mode.reset_index()[['item_id', 'sell_price']]
price_mode = price_mode.rename(columns={'sell_price':'standard_price'})

sell_prices = sell_prices.merge(price_mode, on='item_id', how='left')
sell_prices['discount_rate'] = sell_prices['sell_price'] / sell_prices['standard_price']

price_mean = sell_prices[['item_id', 'sell_price']].groupby('item_id').mean().rename(columns={'sell_price':'sell_price_mean'})
price_mean = pd.DataFrame(price_mean).reset_index()
price_std = sell_prices[['item_id', 'sell_price']].groupby('item_id').std().rename(columns={'sell_price':'sell_price_std'})
price_std = pd.DataFrame(price_std).reset_index()
price_max = sell_prices[['item_id', 'sell_price']].groupby('item_id').max().rename(columns={'sell_price':'sell_price_max'})
price_max = pd.DataFrame(price_max).reset_index()
price_min = sell_prices[['item_id', 'sell_price']].groupby('item_id').min().rename(columns={'sell_price':'sell_price_min'})
price_min = pd.DataFrame(price_min).reset_index()
price_skew = sell_prices[['item_id', 'sell_price']].groupby('item_id').skew().rename(columns={'sell_price':'sell_price_skew'})
price_skew = pd.DataFrame(price_skew).reset_index()

price_info = pd.DataFrame(columns=['item_id'])
price_info = price_info.merge(price_mean, on='item_id', how='outer')
price_info = price_info.merge(price_std, on='item_id', how='inner')
price_info = price_info.merge(price_max, on='item_id', how='inner')
price_info = price_info.merge(price_min, on='item_id', how='inner')
price_info = price_info.merge(price_skew, on='item_id', how='inner')

sell_prices = sell_prices.merge(price_info, on='item_id', how='left')

del price_mean, price_std, price_max, price_min, price_skew, price_info
gc.collect()
train = train.merge(calendar, on='d', how='left')
train = train.merge(sell_prices, on=['store_id', 'item_id', 'wm_yr_wk'], how='left')

del calendar, sell_prices
train = reduce_mem_usage(train)
gc.collect()
tmp = pd.DataFrame(train.groupby(['item_id'])[['sell_price', 'standard_price', 'discount_rate']].apply(lambda x: x.mode()))
tmp = tmp.reset_index()[['item_id', 'sell_price', 'standard_price', 'discount_rate']]
tmp = tmp.rename(columns={'sell_price':'tmp1', 'standard_price':'tmp2', 'discount_rate':'tmp3'})

train = train.merge(tmp, on='item_id', how='left')
train.loc[train['sell_price'].isnull(), 'sell_price']         = train['tmp1']
train.loc[train['standard_price'].isnull(), 'standard_price'] = train['tmp2']
train.loc[train['discount_rate'].isnull(), 'discount_rate']   = train['tmp3']
train.drop(['tmp1', 'tmp2', 'tmp3'], axis=1, inplace=True)

del tmp
train = reduce_mem_usage(train)
gc.collect()
train['snap'] = 0
train.loc[(train['state_id']=='CA')&(train['snap_CA']==1), 'snap'] = 1
train.loc[(train['state_id']=='TX')&(train['snap_TX']==1), 'snap'] = 1
train.loc[(train['state_id']=='WI')&(train['snap_WI']==1), 'snap'] = 1
train['snap'] = LabelEncoder().fit_transform(list(train['snap'].astype(str).values))
train.drop(['snap_CA', 'snap_TX', 'snap_WI'], axis=1, inplace=True)

train = reduce_mem_usage(train)
gc.collect()
train['store_rank'] = 1
train.loc[train['store_id'].isin(['CA_4', 'WI_1']), 'store_rank'] = 0
train.loc[train['store_id'].isin(['CA_1', 'CA_3']), 'store_rank'] = 2

train = reduce_mem_usage(train)
gc.collect()
tmp = train[['id_', 'event_name_1', 'target']].groupby(['id_', 'event_name_1'], as_index=False).mean()

tmp1 = tmp.loc[tmp['event_name_1']=='NoEvent', ['id_', 'target']]
tmp1.columns = ['id_', 'target_no_event']

tmp2 = tmp[tmp['event_name_1']!='NoEvent']
tmp2.columns = ['id_', 'event', 'target_event']

tmp3 = tmp2.merge(tmp1, on='id_', how='left')

tmp3['event_up_lift'] = tmp3['target_event'] / tmp3['target_no_event']

train = train.merge(tmp3[['id_', 'event', 'event_up_lift']], left_on=['id_', 'event_name_1'], right_on=['id_', 'event'], how='left')

train.drop(['event'], axis=1, inplace=True)
train['event_up_lift'].fillna(1, inplace=True)

del tmp, tmp1, tmp2, tmp3
train = reduce_mem_usage(train)
gc.collect()
def ref_past(df, col, base, nlist):
    for i in nlist:
        df[f'{col}_{i}'] = df.groupby([base])[col].apply(lambda x:x.shift(i))
        df = reduce_mem_usage(df)
        gc.collect()
    return df

def agg_past(df, col, base, nlist, windowlist):
    for n in nlist:
        for w in windowlist:
            df[f'{col}_{n}_rolling_{w}_mean'] = df.groupby([base])[f'{col}_{n}'].apply(lambda x: x.rolling(w, 1).mean())
            df[f'{col}_{n}_rolling_{w}_std'] = df.groupby([base])[f'{col}_{n}'].apply(lambda x: x.rolling(w, 1).std())
            df[f'{col}_{n}_rolling_{w}_max'] = df.groupby([base])[f'{col}_{n}'].apply(lambda x: x.rolling(w, 1).max())
            df[f'{col}_{n}_rolling_{w}_min'] = df.groupby([base])[f'{col}_{n}'].apply(lambda x: x.rolling(w, 1).min())
            df[f'{col}_{n}_rolling_{w}_skew'] = df.groupby([base])[f'{col}_{n}'].apply(lambda x: x.rolling(w, 1).skew())
            df[f'{col}_{n}_rolling_{w}_kurt'] = df.groupby([base])[f'{col}_{n}'].apply(lambda x: x.rolling(w, 1).kurt())
            
            df[f'{col}_{n}_ewm_{w}_mean'] = df.groupby([base])[f'{col}_{n}'].apply(lambda x: x.ewm(span=w).mean())
        df = reduce_mem_usage(df)
        gc.collect()
    return df

train = ref_past(train, 'target', 'id_', [28, 29, 30, 56, 180, 365])
train = agg_past(train, 'target', 'id_', [28, 29, 56], [7, 28, 90])


train['target_rolling_28_56_7_mean'] = train['target_28_rolling_7_mean'] / train['target_56_rolling_7_mean']
train['target_rolling_29_56_7_mean'] = train['target_29_rolling_7_mean'] / train['target_56_rolling_7_mean']

train['target_rolling_28_56_28_mean'] = train['target_28_rolling_28_mean'] / train['target_56_rolling_28_mean']
train['target_rolling_29_56_28_mean'] = train['target_29_rolling_28_mean'] / train['target_56_rolling_28_mean']

train['target_rolling_28_56_90_mean'] = train['target_28_rolling_90_mean'] / train['target_56_rolling_90_mean']
train['target_rolling_29_56_90_mean'] = train['target_29_rolling_90_mean'] / train['target_56_rolling_90_mean']


train['target_ewm_28_56_7_mean'] = train['target_28_ewm_7_mean'] / train['target_56_ewm_7_mean']
train['target_ewm_29_56_7_mean'] = train['target_29_ewm_7_mean'] / train['target_56_ewm_7_mean']

train['target_ewm_28_56_28_mean'] = train['target_28_ewm_28_mean'] / train['target_56_ewm_28_mean']
train['target_ewm_29_56_28_mean'] = train['target_29_ewm_28_mean'] / train['target_56_ewm_28_mean']

train['target_ewm_28_56_90_mean'] = train['target_28_ewm_90_mean'] / train['target_56_ewm_90_mean']
train['target_ewm_29_56_90_mean'] = train['target_29_ewm_90_mean'] / train['target_56_ewm_90_mean']


train = reduce_mem_usage(train)
gc.collect()
for w in [7, 28]:
    train[f'sell_price_rolling_{w}_mean'] = train.groupby(['id_'])['sell_price'].apply(lambda x: x.rolling(w, 1).mean())
    train[f'sell_price_rolling_{w}_std'] = train.groupby(['id_'])['sell_price'].apply(lambda x: x.rolling(w, 1).std())
    train[f'sell_price_rolling_{w}_max'] = train.groupby(['id_'])['sell_price'].apply(lambda x: x.rolling(w, 1).max())
    train[f'sell_price_rolling_{w}_min'] = train.groupby(['id_'])['sell_price'].apply(lambda x: x.rolling(w, 1).min())
    train[f'sell_price_rolling_{w}_skew'] = train.groupby(['id_'])['sell_price'].apply(lambda x: x.rolling(w, 1).skew())
    train[f'sell_price_rolling_{w}_kurt'] = train.groupby(['id_'])['sell_price'].apply(lambda x: x.rolling(w, 1).kurt())

    train[f'sell_price_ewm_{w}_mean'] = train.groupby(['id_'])['sell_price'].apply(lambda x: x.ewm(span=w).mean())
    
    train = reduce_mem_usage(train)
for col in train.select_dtypes(include='category').columns.tolist() + train.select_dtypes(include='object').columns.tolist():
    train[col] = LabelEncoder().fit_transform(list(train[col].astype(str).values))
    
train = reduce_mem_usage(train)
gc.collect()
train['state_event_name'] = train['state_id'].astype(str) + '__' + train['event_name_1'].astype(str)
train['state_event_name'] = LabelEncoder().fit_transform(list(train['state_event_name'].astype(str).values))

train['state_event_type'] = train['state_id'].astype(str) + '__' + train['event_type_1'].astype(str)
train['state_event_type'] = LabelEncoder().fit_transform(list(train['state_event_type'].astype(str).values))

train['cat_state']        = train['cat_id'].astype(str) + '__' + train['state_id'].astype(str)
train['cat_state']        = LabelEncoder().fit_transform(list(train['cat_state'].astype(str).values))

train['month_item_id']    = train['month'].astype(str) + '__' + train['item_id'].astype(str)
train['month_item_id']    = LabelEncoder().fit_transform(list(train['month_item_id'].astype(str).values))

train['item_event_name']  = train['item_id'].astype(str) + '__' + train['event_name_1'].astype(str)
train['item_event_name']  = LabelEncoder().fit_transform(list(train['item_event_name'].astype(str).values))

train['item_event_type']  = train['item_id'].astype(str) + '__' + train['event_type_1'].astype(str)
train['item_event_type']  = LabelEncoder().fit_transform(list(train['item_event_type'].astype(str).values))

train = reduce_mem_usage(train)
gc.collect()
train = train[~train['target_365'].isnull()]

train.drop(['id', 'date', 'wm_yr_wk'], axis=1, inplace=True)
for col in train.columns:
    train[col] = replace_inf2nan(train[col])
train = reduce_mem_usage(train)
gc.collect()
# hyper parameter

# train = train[train['part']==0].drop(['part'], axis=1)

# val_d = pd.unique(train['d'])[-28:]

# val = train[train['d'].isin(val_d)]
# train = train[~train['d'].isin(val_d)]

# val.drop(['d'], axis=1, inplace=True)
# train.drop(['d'], axis=1, inplace=True)

# X_val = val.drop(['target'], axis=1)
# y_val = val['target']
# X_train = train.drop(['target'], axis=1)
# y_train = train['target']

# X_val = reduce_mem_usage(X_val)
# X_train = reduce_mem_usage(X_train)

# del train, val
# gc.collect()
test = train[train['part'].isin([1, 2])].drop(['part'], axis=1)
train = train[train['part']==0].drop(['part'], axis=1)

test.drop(['d'], axis=1, inplace=True)
train.drop(['d'], axis=1, inplace=True)

X_test = test.drop(['target', 'd'], axis=1)

X_train = train.drop(['target', 'd'], axis=1)
y_train = train['target']

del train, test
gc.collect()
# def objective(trial):
#     num_leaves = trial.suggest_loguniform('num_leaves', 31, 100)
#     learning_rate = trial.suggest_loguniform('learning_rate', 0.01, 1)
#     n_estimators = trial.suggest_loguniform('n_estimators', 1000, 10000)
#     reg_alpha = trial.suggest_loguniform('reg_alpha', 0.01, 1.0)
#     reg_lambda = trial.suggest_loguniform('reg_lambda', 0.01, 1.0)

#     tu_params = {
#         'num_leaves': int(num_leaves),
#         'learning_rate': learning_rate,
#         'n_estimators': int(n_estimators),
#         'reg_alpha': reg_alpha,
#         'reg_lambda': reg_lambda,
#         # constant
#         'max_depth': -1,
#         'objective': 'regression',
#         'random_state': 42,
#         'n_jobs' : -1
#     }
    
#     lgb_reg = lgb.LGBMRegressor(**tu_params)
#     lgb_reg.fit(X_fit.values, y_fit.values)

#     pred = lgb_reg.predict(X_val.values)

#     return np.sqrt(mean_squared_error(y_val, pred))

# study = optuna.create_study()
# study.optimize(objective, n_trials=10000)
lgb_params = {
        'num_leaves': 56,
        'learning_rate': 0.040440091240782024,
        'n_estimators': 5993,
        'reg_alpha': 0.1906787862387116,
        'reg_lambda': 0.23603397158316272,
        # constant
        'max_depth': -1,
        'objective': 'regression',
        'random_state': 42,
        'n_jobs' : -1
    }

lgb_reg = lgb.LGBMRegressor(**lgb_params)
lgb_reg.fit(train.drop(['target'], axis=1).values, train['target'].values)
submission['target'] = lgb_reg.predict(test.drop(['target'], axis=1).values)
submission = submission.pivot(index='id', columns='d', values='target').reset_index()

sample_submission = reduce_mem_usage(pd.read_csv('../input/sample_submission.csv'))
validation_rows = [row for row in sample_submission['id'] if 'validation' in row] 
validation = submission[submission['id'].isin(validation_rows)].dropna(axis=1)
evaluation = submission[~submission['id'].isin(validation_rows)].dropna(axis=1)

submission_columns = sample_submission.columns
validation.columns = submission_columns
evaluation.columns = submission_columns

final = pd.concat([validation, evaluation])
final = sample_submission[['id']].merge(final, on='id', how='left')
final.to_csv('../submission/submission.csv', index=False)