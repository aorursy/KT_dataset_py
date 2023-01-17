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
import os

import gc

import matplotlib.pyplot as plt

import sklearn

import scipy.sparse 

import lightgbm as lgb

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline 

from sklearn.preprocessing import LabelEncoder, RobustScaler

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

from sklearn.pipeline import Pipeline, make_pipeline

from sklearn.model_selection import KFold

from sklearn.metrics import r2_score

from sklearn.metrics import mean_squared_error

import joblib





import lightgbm

from xgboost import XGBRegressor



pd.set_option('display.max_rows', 600)

pd.set_option('display.max_columns', 50)

sns.set(rc={'figure.figsize':(20, 10)})
item_cat = pd.read_csv('../input/competitive-data-science-predict-future-sales/item_categories.csv')

items = pd.read_csv('../input/competitive-data-science-predict-future-sales/items.csv')

sales = pd.read_csv('../input/competitive-data-science-predict-future-sales/sales_train.csv', parse_dates=['date'], infer_datetime_format=True, dayfirst=True)

shops = pd.read_csv('../input/competitive-data-science-predict-future-sales/shops.csv')

test = pd.read_csv('../input/competitive-data-science-predict-future-sales/test.csv')

sample = pd.read_csv('../input/competitive-data-science-predict-future-sales/sample_submission.csv')
sales.head(5)
# some additional preprocessing of the data for easier analysis

unique_dates = pd.DataFrame({'date': sales['date'].drop_duplicates()})

unique_dates['date_parsed'] = pd.to_datetime(unique_dates.date, format="%d.%m.%Y")

unique_dates['day'] = unique_dates['date_parsed'].apply(lambda d: d.day)

unique_dates['month'] = unique_dates['date_parsed'].apply(lambda d: d.month)

unique_dates['year'] = unique_dates['date_parsed'].apply(lambda d: d.year)



train = sales.merge(unique_dates, on='date').sort_values('date_parsed')


data = train.groupby(['year', 'month']).agg({'item_cnt_day': np.sum}).reset_index().pivot(index='month', columns='year', values='item_cnt_day')

data.plot(figsize=(12, 8))
data = train.groupby(['year', 'month', 'day']).agg({'item_cnt_day': np.sum}).unstack('year')

data.plot(figsize=(12, 8))
sales.head()

sales.dtypes

sales.isnull().sum()

sns.boxplot(x=sales['item_cnt_day']);

sales[sales['item_cnt_day'] > 900]

sales[sales['item_id'] == 11373].sort_values(by='item_cnt_day').tail(10)

sales = sales[sales['item_cnt_day'] <= 1000]

sns.boxplot(sales['item_price'])

sales[sales['item_price']>250000]

sales[sales['item_id'] == 6066]

items[items['item_id']==6066]

item_cat[item_cat['item_category_id'] == 75]

sales = sales[sales['item_price'] < 250000]

items['item_name'].shape[0] == items['item_name'].nunique()

shops.shape[0] == shops['shop_name'].nunique()

shops.head()

clean_shop_names = [shop_name[1:] if shop_name[0]=='!' else shop_name for shop_name in shops['shop_name']]

shops['shop_name'] = clean_shop_names


city_names = [shop_name.split(' ')[0] for shop_name in clean_shop_names]

shops['city'] = city_names

shops['city'] = LabelEncoder().fit_transform(shops['city'])
shops.drop(columns='shop_name', inplace=True)

shops.head()

item_cat['sub_category'] = [word.split('-')[1].strip() if len(word.split('-')) > 1 else 'None' for word in item_cat['item_category_name']]

item_cat['sub_category'] = LabelEncoder().fit_transform(item_cat['sub_category'])
item_cat.drop(columns='item_category_name', inplace=True)

item_cat.head()

items.drop(columns='item_name', inplace=True)

sales.head()

test.head()

from itertools import product



grid = []



for month in sales['date_block_num'].unique():

    

    shop_ids = sales.loc[sales['date_block_num'] == month, 'shop_id'].unique()

    item_ids = sales.loc[sales['date_block_num'] == month, 'item_id'].unique()

    grid.append(np.array(list(product(shop_ids, item_ids, [month]))))
col_names = ['shop_id', 'item_id', 'date_block_num']

grid_df = pd.DataFrame(np.vstack(grid), columns = col_names)
sales_gb = sales.groupby(['date_block_num', 'shop_id', 'item_id'])

# it drops the date column automatically for sum and mean

agg_sales = sales_gb.agg(

    {

     'item_cnt_day':[np.sum]

    }

).fillna(0).clip(0,20)
agg_sales.columns = ['target']

monthly_sales = pd.merge(grid_df, agg_sales, how='left', on=col_names)

monthly_sales['target'] = monthly_sales['target'].fillna(0).clip(0,20)

test_mod = test[['shop_id', 'item_id']].copy()

test_mod['date_block_num'] = 34

test_mod['target'] = np.nan



data = pd.concat([monthly_sales, 

                  test_mod], axis=0)


data = pd.merge(data, items, 

                         how='left', on=['item_id'])
data = pd.merge(data, items, 

                         how='left', on=['item_id'])


data = pd.merge(data, shops, how='left',

                         on=['shop_id'])
data.tail()


def add_target_encoding(data, join_on, name, y_name='target'):

    

    data_agg = data.groupby(join_on).agg({y_name:['mean']})

    data_agg.columns = [name]



    return pd.merge(data, data_agg, on=join_on)
#month

data = add_target_encoding(data,['date_block_num'], 'target_month')

#month-item

data = add_target_encoding(data,['date_block_num', 'item_id'], 

                           'target_month_item')
#month-shop

data = add_target_encoding(data,['date_block_num', 'shop_id'], 

                           'target_month_shop')
data = add_target_encoding(data,['date_block_num', 'shop_id'], 

                           'target_month_shop_category')

small_int_columns = ['city', 'item_category_id', 'category', 'sub_category',

                     'date_block_num', 'shop_id']
target_columns = [col for col in data.columns if col.startswith('target')]



for col in target_columns:

    data[col] = data[col].astype(np.float32)


del sales, test, monthly_sales, test_mod

gc.collect()


def lag_features(df, lags, group_cols, shift_col):

    """

    Arguments:

        df (pd.DataFrame)

        lags (list((int)): the number of months to lag by

        group_cols (list(str)): the list of columns that need to be the merged key

        shift_col (str): the column name that is to be shifted by

    """



    for lag in lags:

        new_col = '{0}_lag_{1}'.format(shift_col, lag)

        df[new_col] = df.groupby(group_cols)[shift_col].shift(lag)



    return df
lags = [1, 2, 3, 6, 12]

group_cols = ['shop_id', 'item_id']

order_col = 'date_block_num'



data = data.sort_values(by=group_cols+[order_col], ascending=True)

data = lag_features(data, lags, group_cols, 'target')
data = lag_features(data, lags, group_cols, 'target_month_item')

data = lag_features(data, lags, group_cols, 'target_month_shop')

one_lag_columns = [col for col in target_columns if col not in ['target', 'target_month_item', 'target_month_shop']]



for col in one_lag_columns:

    data = lag_features(data, [1], group_cols, col)
data = data[data['date_block_num'] >= 12]
y = data[['target', 'date_block_num']]

data.drop(columns=target_columns, inplace=True)

gc.collect()

data = data.fillna(0)

X_train = data.loc[data['date_block_num'] <= 32].drop(columns=['date_block_num'])

X_valid = data.loc[data['date_block_num'] == 33].drop(columns=['date_block_num'])

X_test = data.loc[data['date_block_num'] == 34].drop(columns=['date_block_num'])



y_train = y.loc[y['date_block_num'] <= 32, 'target'].values

y_valid = y.loc[y['date_block_num'] == 33, 'target'].values
lr = LinearRegression()



# remember to scale data when adding regularization

lr.fit(X_train, y_train)
y_valid_predict_lr = lr.predict(X_valid).clip(0, 20)

np.sqrt(np.mean((y_valid_predict_lr - y_valid.clip(0, 20))**2))

lgb_params = { 'max_depth':5,

               'feature_fraction': 0.75,

               'metric': 'rmse',

               'nthread':1, 

               'min_data_in_leaf': 2**7, 

               'bagging_fraction': 0.75, 

               'learning_rate': 0.03, 

               'objective': 'mse', 

               'bagging_seed': 2**7, 

               'num_leaves': 40,

               'bagging_freq':1,

               'verbose':0 

              }


dtrain = lightgbm.Dataset(X_train, label=y_train.clip(0,20))

dvalid = lightgbm.Dataset(X_valid, label=y_valid.clip(0,20))
eval_results = {}

model = lightgbm.train(lgb_params,

                       dtrain,

                       valid_sets=[dtrain,dvalid],

                       valid_names=['train', 'valid'],

                       evals_result=eval_results,

                       num_boost_round=200,

                       early_stopping_rounds=10,

                       verbose_eval=True)

y_valid_predict_lgbm = model.predict(X_valid).clip(0, 20)

np.sqrt(np.mean((y_valid_predict_lgbm - y_valid.clip(0, 20))**2))

dates_train = data.loc[data['date_block_num'] <= 32, 'date_block_num']

dates_valid = data.loc[data['date_block_num'] == 33, 'date_block_num']

dates_test = data.loc[data['date_block_num'] == 34, 'date_block_num']
pred_lr = lr.predict(X_valid.fillna(0)).clip(0,20)

pred_lgb = model.predict(X_valid).clip(0, 20)

X_valid_level2 = np.c_[pred_lr, pred_lgb]

dates_train_level2 = dates_train[dates_train.isin([27, 28, 29, 30, 31, 32])]



y_train_level2 = y_train[dates_train.isin([27, 28, 29, 30, 31, 32])].clip(0,20)
X_train_level2 = np.zeros([y_train_level2.shape[0], 2])

y_train_with_index = pd.Series(data=y_train, index=X_train.index)



for cur_block_num in [27, 28, 29, 30, 31, 32]:

    

    print(cur_block_num)

    

    train_indeces = dates_train[dates_train < cur_block_num].index

    valid_indeces = dates_train[dates_train == cur_block_num].index 

    

    cur_X_train = X_train.loc[train_indeces]

    cur_y_train = y_train_with_index.loc[train_indeces]

    

    lr.fit(cur_X_train.fillna(0), cur_y_train)

    model = lightgbm.train(lgb_params, lightgbm.Dataset(cur_X_train, 

                                              label=cur_y_train), 100)

    

    cur_X_valid = X_train.loc[valid_indeces]

    

     

    X_train_level2[np.where(dates_train_level2==cur_block_num) , 0] = lr.predict(cur_X_valid.fillna(0)).clip(0,20)

    X_train_level2[np.where(dates_train_level2==cur_block_num) , 1] = model.predict(cur_X_valid).clip(0,20)
alphas_to_try = np.linspace(0, 1, 1001)
rmse_scores = [np.sqrt(mean_squared_error(y_train_level2, alpha*X_train_level2[:, 0]+(1-alpha)*X_train_level2[:, 1])) for alpha in alphas_to_try]
best_alpha = alphas_to_try[np.argmin(rmse_scores)]

rmse_train_simple_mix = np.max(rmse_scores)

rmse_train_simple_mix

test = pd.read_csv(os.path.join('../input/competitive-data-science-predict-future-sales/', 'test.csv'))

test.drop(columns='ID', inplace=True)
X_test_final = pd.merge(test, X_test, on=['shop_id', 'item_id'])

y_test_predict_lr = lr.predict(X_test_final.fillna(0)).clip(0, 20)

y_test_predict_lgb = model.predict(X_test_final).clip(0, 20)

y_final_predict = best_alpha*y_test_predict_lr+(1-best_alpha)*y_test_predict_lgb

submission = sample.copy()



submission['item_cnt_month'] = y_final_predict

submission.to_csv('submission_smpl.csv',

                        index = False)