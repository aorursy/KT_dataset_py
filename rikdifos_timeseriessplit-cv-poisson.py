%config InlineBackend.figure_format = 'svg'

%matplotlib inline



import warnings

warnings.filterwarnings('ignore')

import pandas as pd

import numpy as np

import seaborn as sns

pd.set_option('display.max_columns', 500)

pd.set_option('display.max_rows', 500)

import matplotlib.pyplot as plt

import lightgbm as lgb

from datetime import datetime, timedelta

from sklearn import preprocessing, metrics



from sklearn.model_selection import TimeSeriesSplit



import gc

import os

import time



plt.style.use('seaborn')
def reduce_mem_usage(df, verbose=True):

    '''reduce RAM usage

    '''

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024 ** 2    

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





# function to read the data and merge it (ignoring some columns, this is a very fst model)

def read_data():

    '''data input

    '''

    print('Reading files...')

    calendar = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/calendar.csv')

    calendar = reduce_mem_usage(calendar)

    print('Calendar has {} rows and {} columns'.format(calendar.shape[0], calendar.shape[1]))

    sell_prices = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sell_prices.csv')

    sell_prices = reduce_mem_usage(sell_prices)

    print('Sell prices has {} rows and {} columns'.format(sell_prices.shape[0], sell_prices.shape[1]))

    sales_train_evaluation = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sales_train_evaluation.csv')

    print('Sales train evaluation has {} rows and {} columns'.format(sales_train_evaluation.shape[0], sales_train_evaluation.shape[1]))

    submission = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sample_submission.csv')

    return calendar, sell_prices, sales_train_evaluation, submission





def melt_and_merge(calendar, sell_prices, sales_train_evaluation, submission, nrows = 55000000, merge = False):

    

    # melt sales data, get it ready for training

    sales_train_evaluation = pd.melt(sales_train_evaluation, id_vars = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'], var_name = 'day', value_name = 'demand')

    print('Melted sales train validation has {} rows and {} columns'.format(sales_train_evaluation.shape[0], sales_train_evaluation.shape[1]))

    sales_train_evaluation = reduce_mem_usage(sales_train_evaluation)

    

    # seperate test dataframes

    test1_rows = [row for row in submission['id'] if 'validation' in row]

    test2_rows = [row for row in submission['id'] if 'evaluation' in row]

    test1 = submission[submission['id'].isin(test1_rows)]

    test2 = submission[submission['id'].isin(test2_rows)]

    

    # change column names

    test1.columns = ['id', 'd_1914', 'd_1915', 'd_1916', 'd_1917', 'd_1918', 'd_1919', 'd_1920', 'd_1921', 'd_1922', 'd_1923', 'd_1924', 'd_1925', 'd_1926', 'd_1927', 'd_1928', 'd_1929', 'd_1930', 'd_1931', 

                      'd_1932', 'd_1933', 'd_1934', 'd_1935', 'd_1936', 'd_1937', 'd_1938', 'd_1939', 'd_1940', 'd_1941']

    test2.columns = ['id', 'd_1942', 'd_1943', 'd_1944', 'd_1945', 'd_1946', 'd_1947', 'd_1948', 'd_1949', 'd_1950', 'd_1951', 'd_1952', 'd_1953', 'd_1954', 'd_1955', 'd_1956', 'd_1957', 'd_1958', 'd_1959', 

                      'd_1960', 'd_1961', 'd_1962', 'd_1963', 'd_1964', 'd_1965', 'd_1966', 'd_1967', 'd_1968', 'd_1969']

    

    # get product table

    product = sales_train_evaluation[['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']].drop_duplicates()

    

    # merge with product table

    product['id'] = product['id'].str.replace('_evaluation','_validation')

    test1 = test1.merge(product, how = 'left', on = 'id') # validation

    product['id'] = product['id'].str.replace('_validation','_evaluation')

    test2 = test2.merge(product, how = 'left', on = 'id') # evaluation

    

    # 

    test1 = pd.melt(test1, id_vars = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'], var_name = 'day', value_name = 'demand')

    test2 = pd.melt(test2, id_vars = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'], var_name = 'day', value_name = 'demand')

    

    sales_train_evaluation['part'] = 'train'

    test1['part'] = 'validation'

    test2['part'] = 'evaluation'

    

    data = pd.concat([sales_train_evaluation, test1, test2], axis = 0)

    

    del sales_train_evaluation, test1, test2

    

    # get only a sample for fst training

    data = data.loc[nrows:]

    

    # drop some calendar features

    # calendar.drop(['weekday', 'wday', 'month', 'year','snap_CA','snap_TX','snap_WI'], inplace = True, axis = 1)

    

    # delete test2 for now

    data = data[data['part'] != 'validation']

    

    if merge:

        # notebook crash with the entire dataset (maybee use tensorflow, dask, pyspark xD)

        data = pd.merge(data, calendar, how = 'left', left_on = ['day'], right_on = ['d'])

        data.drop(['weekday', 'wday', 'month', 'year','snap_CA','snap_TX','snap_WI'], inplace = True, axis = 1)

        # get the sell price data (this feature should be very important)

        data = data.merge(sell_prices, on = ['store_id', 'item_id', 'wm_yr_wk'], how = 'left')

        print('Our final dataset to train has {} rows and {} columns'.format(data.shape[0], data.shape[1]))

    else: 

        pass

    

    data.to_pickle('data_clean.pkl')

    gc.collect()

    

    return data
cat = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']



def transform(data):

    '''data transformation

    '''

    start = time.time()

    nan_features = ['event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']

    for feature in nan_features:

        data[feature].fillna('unknown', inplace = True)

    cat = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']

    for feature in cat:

        encoder = preprocessing.LabelEncoder()

        data[feature] = encoder.fit_transform(data[feature])

    print('Data transformation costs %7.2f seconds'%(time.time()-start))

    return data



def simple_fe(data):

    '''do some feature engineering

    '''

    start = time.time()

    # rolling demand features

    data_fe = data[['id', 'demand']]

    

    window = 28

    periods = [7, 15, 30, 90]

    group = data_fe.groupby('id')['demand']

    

    # most recent lag data

    for period in periods:

        data_fe['demand_rolling_mean_t' + str(period)] = group.transform(lambda x: x.shift(window).rolling(period).mean())



    periods = [7, 90]

    for period in periods:

        data_fe['demand_rolling_std_t' + str(period)] = group.transform(lambda x: x.shift(window).rolling(period).std())

        

    # reduce memory

    data_fe = reduce_mem_usage(data_fe)

    

    # get time features

    data['date'] = pd.to_datetime(data['date'])

    time_features = ['year', 'month', 'week', 'day', 'dayofweek', 'dayofyear']

    dtype = np.int16

    for time_feature in time_features:

        data[time_feature] = getattr(data['date'].dt, time_feature).astype(dtype)

        

    # concat lag and rolling features with main table

    lag_rolling_features = [col for col in data_fe.columns if col not in ['id', 'demand']]

    data = pd.concat([data, data_fe[lag_rolling_features]], axis = 1)

    

    #data['weekends'] = 0

    #data.loc[(data['dayofweek'] == 5) | (data['dayofweek'] == 6),'weekends'] = 1

    data['weekends'] = np.where((data['date'].dt.dayofweek) < 5, 0, 1)



    del data_fe

    gc.collect()

    

    print('Simple feature engineering costs %7.2f seconds'%(time.time()-start))

    return data



def run_lgb(data):

    '''cross validation

    '''

    start = time.time()

    

    data = data.sort_values('date')

    

    x_train = data[data['date'] <= '2016-05-22']

    y_train = x_train['demand']

    

    test = data[(data['date'] > '2016-05-22')]

    



    del data

    gc.collect()



    params = {

        'boosting_type': 'gbdt',

        'metric': 'rmse',

        'objective': 'poisson', # loss function

        'seed': 225,

        'learning_rate': 0.02,

        'lambda': 0.4, # l2 regularization

        'reg_alpha': 0.4, # l1 regularization

        'max_depth': 5, # max depth of decision trees

        'num_leaves': 64, #  number of leaves

        'bagging_fraction': 0.7, # bootstrap sampling

        'bagging_freq' : 1,

        'colsample_bytree': 0.7 # feature sampling

    }

    

      

    oof = np.zeros(len(x_train))

    preds = np.zeros(len(test))

    

    n_fold = 3 #3 for timely purpose of the kernel

    folds = TimeSeriesSplit(n_splits=n_fold) # use TimeSeriesSplit cv

    splits = folds.split(x_train, y_train)



    feature_importance_df = pd.DataFrame()

    

    for fold, (trn_idx, val_idx) in enumerate(splits):

        print(f'Training fold {fold + 1}')

        

        train_set = lgb.Dataset(x_train.iloc[trn_idx][features], y_train.iloc[trn_idx], categorical_feature = cat)

        

        val_set = lgb.Dataset(x_train.iloc[val_idx][features], y_train.iloc[val_idx], categorical_feature = cat)



        model = lgb.train(params, train_set, num_boost_round = 2400, early_stopping_rounds = 50, 

                          valid_sets = [val_set], verbose_eval = 50)

        

        

        lgb.plot_importance(model, importance_type = 'gain', precision = 0,

                            height = 0.5, figsize = (6, 10), 

                            title = f'fold {fold} feature importance', ignore_zero = True) 

        

        fold_importance_df = pd.DataFrame()

        fold_importance_df['feature'] = features

        fold_importance_df['importance'] = model.feature_importance(importance_type = 'gain')

        fold_importance_df['fold'] = fold + 1

        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)



        oof[val_idx] = model.predict(x_train.iloc[val_idx][features]) # prediction

        preds += model.predict(test[features]) / 3 # calculate mean prediction value of 3 models

        print('-' * 50)

        print('\n')

    model.save_model('model.lgb') # save model

    del x_train

        

    print('3 folds cross-validation costs %7.2f seconds'%(time.time() - start))



    oof_rmse = np.sqrt(metrics.mean_squared_error(y_train, oof))

    print(f'Our out of folds rmse is {oof_rmse}')

    del y_train

        

    test = test[['id', 'date', 'demand']]

    test['demand'] = preds

    gc.collect()

    return test, feature_importance_df





def predict(test, submission):

    '''predict test and validation data label

    '''

    start = time.time()

    predictions = test[['id', 'date', 'demand']]

    predictions = pd.pivot(predictions, index = 'id', columns = 'date', values = 'demand').reset_index()

    predictions.columns = ['id'] + ['F' + str(i + 1) for i in range(28)]

    predictions.to_csv('predictions.csv', index = False)



    prediction_val = predictions.copy()

    prediction_val['id'] = prediction_val['id'].str.replace('_evaluation','_validation') # change id to validation

    prediction_val.to_csv('prediction_val.csv', index = False)

    

    concated = pd.concat([predictions, prediction_val])

    del predictions, prediction_val,

    #final = submission[['id']].merge(concated, on = 'id', how='left')

    #del concated

    print('final dataset to train has {} rows and {} columns'.format(concated.shape[0], concated.shape[1]))

    concated.to_csv('submission.csv', index = False)

    print('Data prediction costs %7.2f seconds'%(time.time() - start))

    



# define list of features



features = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2', 'sell_price', 'year', 

                'month', 'week', 'day', 'dayofweek', 'dayofyear', 'demand_rolling_mean_t7', 'demand_rolling_mean_t15', 'demand_rolling_mean_t30', 'demand_rolling_mean_t90',

                'demand_rolling_std_t7', 'demand_rolling_std_t90','weekends']
def train_and_evaluate(): 

    '''套娃

    '''

    calendar, sell_prices, sales_train_evaluation, submission = read_data()

    data = melt_and_merge(calendar, sell_prices, sales_train_evaluation, submission, nrows = 27500000, merge = True)

    data = transform(data)

    data['date'] = pd.to_datetime(data['date'])

    days = abs((data['date'].min() - data['date'].max()).days)

    # how many training data do we need to train with at least 2 years and consider lags

    need = 365 + 365 + 90 + 28

    print(f'We have {(days - 28)} days of training history')

    print(f'we have {(days - 28 - need)} days left')

    if (days - 28 - need) > 0:

        print('We have enought training data, lets continue')

    else:

        print('Get more training data, training can fail')

        

    data = simple_fe(data)

    

    data = reduce_mem_usage(data)



    print('Removing first 118 days')

    # eliminate the first 118 days of our train data because of lags

    min_date = data['date'].min() + timedelta(days = 118)

    data = data[data['date'] > min_date]

    cat = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']



    test, feature_importance_df = run_lgb(data)

    del data

    fi = (feature_importance_df[['feature', 'importance']]

        .groupby('feature')

        .mean()

        .sort_values(by='importance', ascending=False))

    fi['feature'] = fi.index

    plt.figure(figsize=(6,7))

    sns.barplot(x='importance', y='feature', data=fi[:40])

    plt.title('LightGBM Features (averaged over folds)')

    predict(test, submission) 
%%time

train_and_evaluate() 