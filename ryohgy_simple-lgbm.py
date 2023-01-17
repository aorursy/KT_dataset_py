# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session



import gc
sales_train_validation = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv')

print(sales_train_validation.shape)

calendar = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/calendar.csv')

print(calendar.shape)

sell_prices = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sell_prices.csv')

print(sell_prices.shape)
def convert_categorical_columns(df, columns):

    for column in columns:

        df[column] = df[column].astype('category')

        df[column] = df[column].cat.codes

        df[column] = df[column].astype('category')
sales_train_validation
for i in range(1914, 1914+28):

    sales_train_validation['d_'+str(i)] = 0
convert_categorical_columns(sales_train_validation, ['dept_id', 'cat_id', 'state_id'])
sales_data = sales_train_validation.melt(id_vars=['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'],

                                         var_name='d',

                                         value_name='sales')
df = sales_train_validation.loc[:, sales_train_validation.columns.str.startswith('d_')]

sales_data['lag_7'] = df.shift(7, axis=1).melt()['value']

sales_data['lag_28'] = df.shift(28, axis=1).melt()['value']



del df

gc.collect()
df = sales_train_validation.loc[:, sales_train_validation.columns.str.startswith('d_')]

sales_data['lag_28_rmean_7'] = df.shift(28, axis=1).rolling(7, axis=1).mean().melt()['value']

sales_data['lag_28_rmean_28'] = df.shift(28, axis=1).rolling(28, axis=1).mean().melt()['value']



del df

gc.collect()
df = pd.DataFrame()

for i in range(7):

    d_step7 = ['d_' + str(x) for x in range(1914+28) if (x % 7) == i]

    temp = sales_train_validation.loc[:, sales_train_validation.columns.isin(d_step7)]

    temp = temp.shift(28, axis=1).rolling(4, axis=1).mean()

    temp['id'] = sales_train_validation['id']

    temp = temp.melt(id_vars=['id'], var_name='d', value_name='lag_28_rmean_4_step_7')

    temp['lag_28_rmean_4_step_7'] = temp['lag_28_rmean_4_step_7'].astype(np.float32)

    df = df.append(temp)

sales_data = sales_data.merge(df, how='left', on=['id', 'd'])



del df

gc.collect()
sales_data.head(30)
sales_data.dropna(inplace=True)
sales_data['d'] = sales_data['d'].map(lambda s: int(s[2:]))

sales_data['d'] = sales_data['d'].astype(np.int16)

sales_data['sales'] = sales_data['sales'].astype(np.int16)

sales_data['lag_7'] = sales_data['lag_7'].astype(np.int16)

sales_data['lag_28'] = sales_data['lag_28'].astype(np.int16)

sales_data['lag_28_rmean_7'] = sales_data['lag_28_rmean_7'].astype(np.float32)

sales_data['lag_28_rmean_28'] = sales_data['lag_28_rmean_28'].astype(np.float32)
sales_data.dtypes
calendar
convert_categorical_columns(calendar, ['event_name_1', 'event_type_1', 'event_name_2', 'event_type_2'])
calendar['day'] = calendar['date'].map(lambda s: s[-2:])

calendar['is_weekend'] = calendar['weekday'].isin(['Saturday', 'Sunday'])
calendar.drop(['date', 'weekday'], axis=1, inplace=True)
calendar['d'] = calendar['d'].map(lambda s: int(s[2:]))

calendar['wm_yr_wk'] = calendar['wm_yr_wk'].astype(np.int16)

calendar['wday'] = calendar['wday'].astype(np.int8)

calendar['month'] = calendar['month'].astype(np.int8)

calendar['year'] = calendar['year'].astype(np.int16)

calendar['d'] = calendar['d'].astype(np.int16)

calendar['snap_CA'] = calendar['snap_CA'].astype(np.int8)

calendar['snap_TX'] = calendar['snap_TX'].astype(np.int8)

calendar['snap_WI'] = calendar['snap_WI'].astype(np.int8)

calendar['day'] = calendar['day'].astype(np.int8)
calendar
sales_data = sales_data.merge(calendar, on='d', how='left')
sell_prices
sell_prices['sell_price'] = sell_prices['sell_price'].astype(np.float32)
sales_data = sales_data.merge(sell_prices, how='left', on=['store_id', 'item_id', 'wm_yr_wk'])
sales_data.dropna(inplace=True)
convert_categorical_columns(sales_data, ['item_id', 'store_id'])
sales_data.set_index('id', inplace=True)
sales_data.head(30)
del sales_train_validation, calendar, sell_prices

gc.collect()
MIN_D = 1400

sales_data = sales_data[sales_data['d'] > MIN_D]
d_lgb_train = [i for i in range(1914-28)]

d_lgb_eval  = [i for i in range(1914-28, 1914)]

d_test_1w   = [i for i in range(1914, 1914+7)]

d_test_2w   = [i for i in range(1914+7, 1914+7+7)]

d_test_3w   = [i for i in range(1914+7+7, 1914+7+7+7)]

d_test_4w   = [i for i in range(1914+7+7+7, 1914+7+7+7+7)]
lgb_train_x = sales_data.loc[sales_data['d'].isin(d_lgb_train)].drop(['sales'], axis=1)

lgb_train_y = sales_data.loc[sales_data['d'].isin(d_lgb_train)]['sales']

lgb_eval_x = sales_data.loc[sales_data['d'].isin(d_lgb_eval)].drop(['sales'], axis=1)

lgb_eval_y = sales_data.loc[sales_data['d'].isin(d_lgb_eval)]['sales']

test_x_1w = sales_data.loc[sales_data['d'].isin(d_test_1w)].drop(['sales'], axis=1)

test_x_2w = sales_data.loc[sales_data['d'].isin(d_test_2w)].drop(['sales'], axis=1)

test_x_3w = sales_data.loc[sales_data['d'].isin(d_test_3w)].drop(['sales'], axis=1)

test_x_4w = sales_data.loc[sales_data['d'].isin(d_test_4w)].drop(['sales'], axis=1)



del sales_data

gc.collect()
import lightgbm as lgb



lgb_params = {

    'objective': 'poisson',

    'metric': 'rmse',

    'seed': 101,

    'learning_rate': 0.01,

}
lgb_train = lgb.Dataset(lgb_train_x, lgb_train_y)

lgb_eval  = lgb.Dataset(lgb_eval_x, lgb_eval_y, reference=lgb_train)



model = lgb.train(lgb_params, lgb_train, num_boost_round=10000, valid_sets=lgb_eval, early_stopping_rounds=200)
model.save_model('model.txt')
pred_y_1w = model.predict(test_x_1w, num_iteration=model.best_iteration)

result_1w = pd.DataFrame({'d': test_x_1w['d'], 'sales': pred_y_1w})



test_x_2w['lag_7'] = pred_y_1w

pred_y_2w = model.predict(test_x_2w, num_iteration=model.best_iteration)

result_2w = pd.DataFrame({'d': test_x_2w['d'], 'sales': pred_y_2w})



test_x_3w['lag_7'] = pred_y_2w

pred_y_3w = model.predict(test_x_3w, num_iteration=model.best_iteration)

result_3w = pd.DataFrame({'d': test_x_3w['d'], 'sales': pred_y_3w})



test_x_4w['lag_7'] = pred_y_3w

pred_y_4w = model.predict(test_x_4w, num_iteration=model.best_iteration)

result_4w = pd.DataFrame({'d': test_x_4w['d'], 'sales': pred_y_4w})
result = pd.concat([result_1w, result_2w, result_3w, result_4w])

result['d'] = result['d'].map(lambda x: 'F' + str(x - 1913))

result = result.reset_index().pivot(index='id', columns='d', values='sales')

result
result_e = pd.DataFrame(0, index=result.index, columns=result.columns)

result_e.index = pd.Series(result_e.index).apply(lambda s: s.replace('validation', 'evaluation'))



pd.concat([result, result_e]).to_csv('submission.csv')