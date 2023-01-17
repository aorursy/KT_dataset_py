import warnings

warnings.filterwarnings('ignore')

import pandas as pd

import numpy as np

import gc

pd.set_option('display.max_columns', 500)

pd.set_option('display.max_rows', 500)
def reduce_memory(df):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

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

    return df
calendar = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/calendar.csv')

calendar = reduce_memory(calendar)

sales_train_validation = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv')

sales_train_validation = reduce_memory(sales_train_validation)

submission = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sample_submission.csv')
valid_rows = [row for row in submission['id'] if 'validation' in row]

validation = submission.query("id in @valid_rows")

validation.columns = ['id', 'd_1914', 'd_1915', 'd_1916', 'd_1917', 'd_1918', 'd_1919', 'd_1920', 'd_1921', 'd_1922', 'd_1923', 'd_1924', 'd_1925', 'd_1926', 'd_1927', 'd_1928', 'd_1929', 'd_1930', 'd_1931', 

                      'd_1932', 'd_1933', 'd_1934', 'd_1935', 'd_1936', 'd_1937', 'd_1938', 'd_1939', 'd_1940', 'd_1941']

product = sales_train_validation[['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']].drop_duplicates()

validation = validation.merge(product, how = 'left', on = 'id')

validation = pd.melt(validation, id_vars = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'], var_name = 'day', value_name = 'demand')

sales_train_validation = pd.melt(sales_train_validation, id_vars = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'], var_name = 'day', value_name = 'demand')

data = pd.concat([sales_train_validation, validation], axis = 0)

calendar.drop(['wm_yr_wk','weekday', 'wday', 'month', 'year','event_name_1','event_type_1','event_name_2','event_type_2','snap_CA','snap_TX','snap_WI'], inplace = True, axis = 1)

data = pd.merge(data, calendar, how = 'left', left_on = ['day'], right_on = ['d'])

del sales_train_validation, calendar

gc.collect()

data.drop(['d', 'day'], inplace = True, axis = 1)

data['date'] = pd.to_datetime(data['date'])
data_new = data[(data['date'] > '2016-02-28') & (data['date'] <= '2016-04-24')][['id','date','demand']]

data_old = data[(data['date'] <= '2016-02-28')][['id','date','demand']]

validation = data[(data['date'] > '2016-04-24')][['id','date','demand']]
del data

gc.collect()
number_of_product_new = data_new.groupby('id')['demand'].sum().reset_index()

number_of_product_old = data_old.groupby('id')['demand'].sum().reset_index()

number_of_product_new.columns = ['id','demand_new']

number_of_product_old.columns = ['id','demand_old']

number_of_product = pd.merge(number_of_product_old, number_of_product_new, how='inner', left_on=['id'], right_on=['id'])

number_of_product['diff'] = number_of_product['demand_new'] - number_of_product['demand_old']

number_of_product = number_of_product[(number_of_product['diff'] >= 0)]['id']
len(number_of_product)
data_new['weekday'] = data_new['date'].dt.weekday

data_predict_median = data_new.groupby(['id','weekday'])['demand'].median().to_dict()
def predict_median(x):

    return data_predict_median[(x[0], x[1])]
validation['weekday'] = validation['date'].dt.weekday

validation.query("id in @number_of_product", inplace=True)

validation['demand'] = validation[['id','weekday']].apply(lambda x: predict_median(x), axis=1)

validation['demand'].fillna(0,inplace=True)

predictions = validation[['id', 'date', 'demand']]

predictions = pd.pivot(predictions, index = 'id', columns = 'date', values = 'demand').reset_index()

predictions.columns = ['id'] + ['F' + str(i + 1) for i in range(28)]
submission_lightGBM = pd.read_csv('/kaggle/input/lightgbm/submission_lightGBM.csv')

submission = pd.concat([predictions, submission_lightGBM]).drop_duplicates(subset ="id", keep = 'first') 

submission.to_csv('submission.csv', index=False, header=True)