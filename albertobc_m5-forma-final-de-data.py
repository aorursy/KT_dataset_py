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

def rolling_demand_fe(data):
    
    # rolling demand features
    #data['lag_t28'] = data.groupby(['id'])['demand'].transform(lambda x: x.shift(28))
    #data['lag_t29'] = data.groupby(['id'])['demand'].transform(lambda x: x.shift(29))
    #data['lag_t30'] = data.groupby(['id'])['demand'].transform(lambda x: x.shift(30))
    data['rolling_mean_t7'] = data.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(7).mean())
    data['rolling_mean_t30'] = data.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(30).mean())
    data['rolling_mean_t60'] = data.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(60).mean())
    data['rolling_mean_t90'] = data.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(90).mean())
    data['rolling_mean_t180'] = data.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(180).mean())
    #data['rolling_std_t30'] = data.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(30).std())
    
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
def readData(submission_only=False,PATH='/kaggle/input/'):
    import pandas as pd
    print('Reading files...')
    submission = pd.read_csv(PATH+'m5-forecasting-accuracy/sample_submission.csv')
    if submission_only:
        return submission
    else:
        calendar = pd.read_csv(PATH+'m5-forecasting-accuracy/calendar.csv')
        calendar = reduce_mem_usage(calendar)
        print('Calendar has {} rows and {} columns'.format(calendar.shape[0], calendar.shape[1]))

        sell_prices = pd.read_csv(PATH+'m5-forecasting-accuracy/sell_prices.csv')
        sell_prices = reduce_mem_usage(sell_prices)
        print('Sell prices has {} rows and {} columns'.format(sell_prices.shape[0], sell_prices.shape[1]))

        sales_train_validation = pd.read_csv(PATH+'m5-forecasting-accuracy/sales_train_validation.csv')
        print('Sales train validation has {} rows and {} columns'.format(sales_train_validation.shape[0], sales_train_validation.shape[1]))

    return calendar, sell_prices, sales_train_validation, submission
# process the data to get it into a tabular format; pd.melt is especially useful to 'unpack' the target variable (demand)
def melt_and_merge(nrows=5.5e7):
    
    calendar, sell_prices, sales_train_validation, submission = readData()
    
    # drop some calendar features
    calendar.drop(['weekday', 'wday', 'month', 'year'], inplace = True, axis = 1)
    
    # melt sales data, get it ready for training
    sales_train_validation = pd.melt(sales_train_validation, id_vars = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'], 
                                     var_name = 'day', value_name = 'demand')
    
    #print('Melted sales train validation has {} rows and {} columns'.format(sales_train_validation.shape[0], sales_train_validation.shape[1]))
    sales_train_validation = reduce_mem_usage(sales_train_validation)
    
    # seperate test dataframes
    test1_rows = [row for row in submission['id'] if 'validation' in row]
    #test2_rows = [row for row in submission['id'] if 'evaluation' in row]
    test1 = submission[submission['id'].isin(test1_rows)]
    #test2 = submission[submission['id'].isin(test2_rows)]
    
    # change column names
    test1.columns = ['id'] + ['d_{}'.format(i) for i in range(1914,1942)]
    #test2.columns = ['id'] + ['d_{}'.format(i) for i in range(1942,1970)]


    # get product table
    product = sales_train_validation[['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']].drop_duplicates()
    
    # merge with product table
    #test2['id'] = test2['id'].str.replace('_evaluation','_validation')
    test1 = test1.merge(product, how = 'left', on = 'id')
    #test2 = test2.merge(product, how = 'left', on = 'id')
    #test2['id'] = test2['id'].str.replace('_validation','_evaluation')
    
    # 
    test1 = pd.melt(test1, id_vars = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'], var_name = 'day',
                    value_name = 'demand')
    #test2 = pd.melt(test2, id_vars = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'], var_name = 'day',
    #                value_name = 'demand')
    
    sales_train_validation = pd.concat([sales_train_validation, test1], axis = 0) # include test2 later
    
    del test1#, test2
    gc.collect()
    
    # delete first entries otherwise memory errors
    sales_train_validation = sales_train_validation.loc[nrows:]
    
    # delete test2 for now
    #data = data[data['part'] != 'test2']
    
    sales_train_validation = pd.merge(sales_train_validation, calendar, how = 'left', left_on = ['day'], right_on = ['d'])
    sales_train_validation.drop(['d', 'day'], inplace = True, axis = 1)
    
    # get the sell price data (this feature should be very important)
    sales_train_validation = sales_train_validation.merge(sell_prices, on = ['store_id', 'item_id', 'wm_yr_wk'], how = 'left')
    print('Our final dataset to train has {} rows and {} columns'.format(sales_train_validation.shape[0], sales_train_validation.shape[1]))
    
    del calendar, sell_prices; gc.collect();
    
    return sales_train_validation
def append_predictions(test, submission, name):
    predictions = test[['id', 'date', 'demand']]
    predictions = pd.pivot(predictions, index = 'id', columns = 'date', values = 'demand').reset_index()
    predictions.columns = ['id'] + ['F' + str(i + 1) for i in range(28)]

    evaluation_rows = [row for row in submission['id'] if 'evaluation' in row] 
    evaluation = submission[submission['id'].isin(evaluation_rows)]

    validation = submission[['id']].merge(predictions, on = 'id')
    final = pd.concat([validation, evaluation])
    final.to_csv(name, index = False)
    
def generate_predictions(model, features, test, submission, name):

    y_pred = model.predict(test[features])
    test['demand'] = y_pred
    
    append_predictions(test, submission, name)
def transform(data):
    from sklearn.preprocessing import LabelEncoder
    # convert to datetime object
    data['date'] = pd.to_datetime(data.date)
    
    # fill NaN features with unknown
    nan_features = ['event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']
    for feature in nan_features:
        data[feature].fillna('unknown', inplace = True)
    
    # Encode categorical features
    cat = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']
    for feature in cat:
        encoder = LabelEncoder()
        data[feature] = encoder.fit_transform(data[feature])
    
    return data

def transform_stage2(data):
        
    # fill NaN features with unknown
    nan_features = ['lag_t28', 'lag_t29', 'lag_t30', 'rolling_mean_t7', 'rolling_std_t7', 'rolling_mean_t30',
                    'rolling_mean_t90','rolling_mean_t180', 'rolling_std_t30', 'rolling_skew_t30', 'rolling_kurt_t30']
                    
    for feature in nan_features:
        data[feature].fillna('unknown', inplace = True)
    
    return data
data = melt_and_merge(nrows=2.75e7)
data = transform(data)
submission = readData(submission_only=True)
data = time_fe(data)
data = reduce_mem_usage(data)

data = rolling_demande_fe(data)
data = reduce_mem_usage(data)

data = transform_stage2(data)
data = reduce_mem_usage(data)
x_train = data[data['date'] <= '2016-03-27']
y_train = x_train['demand']
x_val = data[(data['date'] > '2016-03-27') & (data['date'] <= '2016-04-24')]
y_val = x_val['demand']
test = data[(data['date'] > '2016-04-24')]
del data

features = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'year', 'month', 'week', 'day', 
            'dayofweek', 'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2', 
            'snap_CA', 'snap_TX', 'snap_WI', 'sell_price', 'lag_t28', 'lag_t29', 'lag_t30', 
            'rolling_mean_t7', 'rolling_std_t7', 'rolling_mean_t30', 'rolling_mean_t90', 
            'rolling_mean_t180', 'rolling_std_t30', 'rolling_skew_t30', 'rolling_kurt_t30']
train_set = lgb.Dataset(x_train[features], y_train)
val_set = lgb.Dataset(x_val[features], y_val)

num_round = 1000
model = lgb.train(params, train_set, valid_sets=[val_set], early_stopping_rounds=10, verbose_eval=200)
name = 'submission_LGBM7.csv'

generate_predictions(model, features, x_val, submission, name)
from sklearn.metrics import mean_squared_error

val_pred = model.predict(x_val[features])
val_score = np.sqrt(mean_squared_error(val_pred, y_val))
print(f'Our val rmse score is {val_score}')