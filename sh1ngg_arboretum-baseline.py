import warnings

warnings.filterwarnings('ignore')

import pandas as pd

import numpy as np

pd.set_option('display.max_columns', 500)

pd.set_option('display.max_rows', 5000)

import matplotlib.pyplot as plt

import lightgbm as lgb

from sklearn import preprocessing, metrics

import gc

import os



def on_kaggle():

    return "KAGGLE_KERNEL_RUN_TYPE" in os.environ

data_folder =  "/kaggle/input/m5-forecasting-accuracy" if on_kaggle() else 'm5-forecasting-accuracy'



for dirname, _, filenames in os.walk(data_folder):

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



def process_cat(data, features):

    for feature in features:

        print('fillna labelEncoding', feature)

        data[feature].fillna('unknown', inplace = True)

        encoder = preprocessing.LabelEncoder()

        data[feature] = encoder.fit_transform(data[feature]).astype(np.int32)

    

    return data

    





def read_data():

    print('Reading files...')

    calendar = pd.read_csv(data_folder + '/calendar.csv',parse_dates=['date'], dtype={ 'wm_yr_wk':np.int32, 'weekday':'category', 'wday':np.int32, 'month':np.int32, 

                                                                 'year':np.int32, 'd':'str', 'event_name_1':'str', 'event_type_1':'str', 

                                                                 'event_name_2':'str','event_type_2':'str', 'snap_CA':np.float32, 'snap_TX':np.float32, 'snap_WI':np.float32})

    print('Calendar has {} rows and {} columns'.format(calendar.shape[0], calendar.shape[1]))

    sell_prices = pd.read_csv(data_folder + '/sell_prices.csv', dtype={'store_id':'str', 'item_id':'str', 'wm_yr_wk':np.int32, 'sell_price':np.float32})

    print('Sell prices has {} rows and {} columns'.format(sell_prices.shape[0], sell_prices.shape[1]))

    sales_train_validation = pd.read_csv(data_folder + '/sales_train_validation.csv')

    print('Sales train validation has {} rows and {} columns'.format(sales_train_validation.shape[0], sales_train_validation.shape[1]))

    submission = pd.read_csv(data_folder + '/sample_submission.csv')

    return calendar, sell_prices, sales_train_validation, submission





def melt_and_merge(calendar, sell_prices, sales_train_validation, submission, merge = False):

    

    # melt sales data, get it ready for training

    sales_train_validation = pd.melt(sales_train_validation, id_vars = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'], var_name = 'day', value_name = 'demand')

    print('Melted sales train validation has {} rows and {} columns'.format(sales_train_validation.shape[0], sales_train_validation.shape[1]))

    sales_train_validation = reduce_mem_usage(sales_train_validation)

    print(sales_train_validation.shape)

    

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

    product = sales_train_validation[['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']].drop_duplicates()

    

    # merge with product table

    test2['id'] = test2['id'].str.replace('_evaluation','_validation')

    test1 = test1.merge(product, how = 'left', on = 'id')

    test2 = test2.merge(product, how = 'left', on = 'id')

    test2['id'] = test2['id'].str.replace('_validation','_evaluation')

    

    # 

    test1 = pd.melt(test1, id_vars = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'], var_name = 'day', value_name = 'demand')

    test2 = pd.melt(test2, id_vars = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'], var_name = 'day', value_name = 'demand')

    

    sales_train_validation['part'] = 0

    test1['part'] = 1

    test2['part'] = 2

    

    data = pd.concat([sales_train_validation, test1, test2], axis = 0)

    data['demand'] = data['demand'].astype(np.float32) 

    

    del sales_train_validation, test1, test2

    gc.collect()

    

    

    # drop some calendar features

    calendar.drop(['weekday', 'wday', 'month', 'year'], inplace = True, axis = 1)

    

    # delete test2 for now

    data = data[data['part'] != 2]

    

    print('merging....')

    

    if merge:

        data = reduce_mem_usage(data)

        # notebook crash with the entire dataset (maybee use tensorflow, dask, pyspark xD)

        data = pd.merge(data, calendar, how = 'left', left_on = ['day'], right_on = ['d'])

        data.drop(['d', 'day', 'part'], inplace = True, axis = 1)

        print('merged with calendar')

        gc.collect()

        print(data.dtypes)

        print(sell_prices.dtypes)

        # get the sell price data (this feature should be very important)

        data = data.merge(sell_prices, on = ['store_id', 'item_id', 'wm_yr_wk'], how = 'left')

        print('Our final dataset to train has {} rows and {} columns'.format(data.shape[0], data.shape[1]))

    else: 

        pass

    

    gc.collect()

    

    return data

        

calendar, sell_prices, sales_train_validation, submission = read_data()

calendar = process_cat(calendar, ['event_name_1', 'event_type_1', 'event_name_2', 'event_type_2'])

sales_train_validation = process_cat(sales_train_validation, ['state_id', 'dept_id', 'cat_id'])

data = melt_and_merge(calendar, sell_prices, sales_train_validation, submission, merge = True)

del calendar, sell_prices, sales_train_validation

gc.collect()
!pip install arboretum