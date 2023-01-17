import warnings

warnings.filterwarnings('ignore')

import pandas as pd

import numpy as np

pd.set_option('display.max_columns', 500)

pd.set_option('display.max_rows', 500)

import matplotlib.pyplot as plt

import seaborn as sns

import lightgbm as lgb

from sklearn import preprocessing, metrics

import gc

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
def reduce_mem_usage(df, verbose=True):

    

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024**2    

    for col in df.columns:

        col_type = df[col].dtypes

        if col_type in numerics:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if   c_min > np.iinfo(np.int8 ).min and c_max < np.iinfo(np.int8).max :

                    df[col] = df[col].astype(np.int8 )

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if   c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)    

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

        

    return df
def read_data():

    

    calendar               = reduce_mem_usage(pd.read_csv('../input/m5-forecasting-accuracy/calendar.csv'))

    sell_prices            = reduce_mem_usage(pd.read_csv('../input/m5-forecasting-accuracy/sell_prices.csv'))    

    sales_train_validation = reduce_mem_usage(pd.read_csv('../input/m5-forecasting-accuracy/sales_train_validation.csv'))

    submission             = reduce_mem_usage(pd.read_csv('../input/m5-forecasting-accuracy/sample_submission.csv'))

    

    print('calendar               has {} rows, {} columns'.format(calendar.shape[0],               calendar.shape[1]))

    print('sell_prices            has {} rows, {} columns'.format(sell_prices.shape[0],            sell_prices.shape[1]))    

    print('sales_train_validation has {} rows, {} columns'.format(sales_train_validation.shape[0], sales_train_validation.shape[1]))

    print('submission             has {} rows, {} columns'.format(submission.shape[0],             submission.shape[1]))

    

    return calendar, sell_prices, sales_train_validation, submission
def melt_and_merge(calendar, sell_prices, sales_train_validation, submission, nrows=55_000_000, merge=False):

    

    

    # sales_train_validation

    sales_train_validation = pd.melt(sales_train_validation,

                                    id_vars   = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'],

                                    var_name  = 'day',

                                    value_name = 'demand')

    print('melted sales_train_validation has {} rows, {} columns'.format(sales_train_validation.shape[0], sales_train_validation.shape[1]))

    

    

    # submission

    test1 = submission[submission['id'].str.contains('validation')]

    test2 = submission[submission['id'].str.contains('evaluation')]

    

    test1.columns = ['id'] + ['d_' + str(i) for i in range(1914, 1942)]

    test2.columns = ['id'] + ['d_' + str(i) for i in range(1942, 1970)]  

    

    product = sales_train_validation[['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']].drop_duplicates()

    test1 = test1.merge(product, on = 'id', how = 'left')

    test2['id'] = test2['id'].str.replace('_evaluation','_validation')

    test2 = test2.merge(product, on = 'id', how = 'left')

    test2['id'] = test2['id'].str.replace('_validation','_evaluation')

    

    test1 = pd.melt(test1,

                    id_vars   = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'],

                    var_name  = 'day',

                    value_name = 'demand')

    test2 = pd.melt(test2,

                    id_vars   = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'],

                    var_name  = 'day',

                    value_name = 'demand') 

    

    sales_train_validation['part'] = 'train'

    test1['part'] = 'test1'

    test2['part'] = 'test2'

    

    data = pd.concat([sales_train_validation, test1, test2], axis = 0).loc[nrows:]

    data = data[data['part'] != 'test2']

    del sales_train_validation, test1, test2, submission

    

    

    # calendar, sell_prices

    if merge:

        calendar = calendar.drop(columns = ['weekday', 'wday', 'month', 'year'])

        data = data.merge(calendar, how = 'left', left_on = ['day'], right_on = ['d']).drop(columns=['d', 'day'])

        data = data.merge(sell_prices, on = ['store_id', 'item_id', 'wm_yr_wk'], how = 'left')

        print('final dataset has {} rows and {} columns'.format(data.shape[0], data.shape[1]))

    

    gc.collect()

    

    return data
calendar, sell_prices, sales_train_validation, submission = read_data()

data = melt_and_merge(calendar, sell_prices, sales_train_validation, submission, nrows = 27_500_000, merge = True)
data.head()
def transform(data):

    

    nan_features = ['event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']

    for feature in nan_features:

        data[feature] = data[feature].fillna('unknown')

    

    cat_features = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']

    for feature in cat_features:

        le = preprocessing.LabelEncoder()

        data[feature] = le.fit_transform(data[feature])

    

    return data
def simple_fe(data):

    

    # demand features

    data['lag_t28']                = data.groupby(['id'])['demand'].transform(lambda x: x.shift(28))

    data['lag_t29']                = data.groupby(['id'])['demand'].transform(lambda x: x.shift(29))

    data['lag_t30']                = data.groupby(['id'])['demand'].transform(lambda x: x.shift(30))

    data['rolling_mean_t7']        = data.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(7).mean())

    data['rolling_std_t7']         = data.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(7).std())

    data['rolling_mean_t30']       = data.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(30).mean())

    data['rolling_std_t30']        = data.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(30).std())

    data['rolling_mean_t90']       = data.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(90).mean())

    data['rolling_std_t90']        = data.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(90).std())

    data['rolling_mean_t180']      = data.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(180).mean())

    data['rolling_std_t180']        = data.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(180).std())

    data['rolling_skew_t30']       = data.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(30).skew())

    data['rolling_kurt_t30']       = data.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(30).kurt())

    

    # price features

    data['lag_price_t1']           = data.groupby(['id'])['sell_price'].transform(lambda x: x.shift(1))

    data['price_change_t1']        = (data['lag_price_t1'] - data['sell_price']) / data['lag_price_t1']

    data['rolling_price_max_t365'] = data.groupby(['id'])['sell_price'].transform(lambda x: x.shift(1).rolling(365).max())

    data['price_change_t365']      = (data['rolling_price_max_t365'] - data['sell_price']) / data['rolling_price_max_t365']

    data['rolling_price_std_t7']   = data.groupby(['id'])['sell_price'].transform(lambda x: x.rolling(7).std())

    data['rolling_price_std_t30']  = data.groupby(['id'])['sell_price'].transform(lambda x: x.rolling(30).std())

    data = data.drop(columns = ['rolling_price_max_t365', 'lag_price_t1'])

    

    # time features

    data['date']      = pd.to_datetime(data['date']) 

    data['year']      = data['date'].dt.year

    data['month']     = data['date'].dt.month

    data['week']      = data['date'].dt.week

    data['day']       = data['date'].dt.day

    data['dayofweek'] = data['date'].dt.dayofweek

    

    return data
data = transform(data)

data = simple_fe(data)

data = reduce_mem_usage(data)
data.head()
def run_lgb(data):

    

    features = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 

                'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2', 'snap_CA', 'snap_TX', 'snap_WI', 'sell_price', 

                'lag_t28', 'lag_t29', 'lag_t30', 'rolling_mean_t7', 'rolling_std_t7', 'rolling_mean_t30', 'rolling_std_t30', 'rolling_mean_t90', 'rolling_std_t90', 'rolling_mean_t180', 'rolling_std_t180', 'rolling_skew_t30', 'rolling_kurt_t30',

                'price_change_t1', 'price_change_t365', 'rolling_price_std_t7', 'rolling_price_std_t30', 

                'year', 'month', 'week', 'day', 'dayofweek']

    

    x_train = data[data['date'] <= '2016-03-27'][features]

    y_train = data[data['date'] <= '2016-03-27']['demand']

    x_val   = data[(data['date'] > '2016-03-27') & (data['date'] <= '2016-04-24')][features]

    y_val   = data[(data['date'] > '2016-03-27') & (data['date'] <= '2016-04-24')]['demand']

    test    = data[(data['date'] > '2016-04-24')]

    

    del data

    gc.collect()    

    

    train_set = lgb.Dataset(x_train, y_train)

    val_set   = lgb.Dataset(x_val, y_val)

    

    params = {

        'boosting_type': 'gbdt',

        'metric': 'rmse',

        'objective': 'regression',

        'n_jobs': -1,

        'seed': 236,

        'learning_rate': 0.01,

        'bagging_fraction': 0.75,

        'bagging_freq': 10, 

        'colsample_bytree': 0.75}

    

    model       = lgb.train(params, train_set, num_boost_round=2500, early_stopping_rounds=50, valid_sets = [train_set, val_set], verbose_eval=100)

    train_pred  = model.predict(x_train)

    train_score = np.sqrt(metrics.mean_squared_error(train_pred, y_train))

    val_pred    = model.predict(x_val)

    val_score   = np.sqrt(metrics.mean_squared_error(val_pred, y_val))

    print(f'train rmse score is {train_score}')

    print(f'val rmse score is {val_score}')

    

    y_pred = model.predict(test[features])

    test['demand'] = y_pred

    

    return test
test = run_lgb(data)
def predict(test, submission):

    predictions = test[['id', 'date', 'demand']]

    predictions = pd.pivot(predictions, index = 'id', columns = 'date', values = 'demand').reset_index()

    predictions.columns = ['id'] + ['F' + str(i + 1) for i in range(28)]



    evaluation_rows = [row for row in submission['id'] if 'evaluation' in row] 

    evaluation = submission[submission['id'].isin(evaluation_rows)]



    validation = submission[['id']].merge(predictions, on = 'id')

    final = pd.concat([validation, evaluation])

    final.to_csv('submission.csv', index = False)
predict(test, submission)