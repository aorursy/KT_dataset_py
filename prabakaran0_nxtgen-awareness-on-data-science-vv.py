# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import matplotlib as mpl

#import matplotlib.pyplot as plt % matplotlib inline

import seaborn as sns

import datetime

import gc

from itertools import product

import lightgbm as lgb

from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import LabelEncoder

import time

from statsmodels.tsa.stattools import acf

data_path = '/kaggle/input/competitive-data-science-predict-future-sales/'



def downcast_dtypes(df):

    start_size = df.memory_usage(deep = True).sum() / 1024**2

    print('Memory usage: {:.2f} MB'.format(start_size))



    float_cols = [c for c in df if df[c].dtype == "float64"]

    int_cols = [c for c in df if df[c].dtype in ["int64", "int32"]]

    df[float_cols] = df[float_cols].astype(np.float32)

    df[int_cols] = df[int_cols].astype(np.int32)

    end_size = df.memory_usage(deep = True).sum() / 1024**2

    print('New Memory usage: {:.2f} MB'.format(end_size))

    return df



def create_record_for_features(df, attrs, target, time_col, aggfunc = np.sum, fill = 0):

    target_for_attrs = df.pivot_table(index = attrs,

                                   values = target, 

                                   columns = time_col, 

                                   aggfunc = aggfunc, 

                                   fill_value = fill,

                                  ).reset_index()

    target_for_attrs.columns = target_for_attrs.columns.map(str)

    target_for_attrs = target_for_attrs.reset_index(drop = True).rename_axis(None, axis = 1)

    return target_for_attrs



def display_df_info(df, name):

    print('-----------Shape of '+ name + '-------------')

    print(df.shape)

    print('-----------Missing values---------')

    print(df.isnull().sum())

    print('-----------Null values------------')

    print(df.isna().sum())

    print('-----------Data types-------------')

    print(df.dtypes)

    print('-----------Memory usage (MB)------')

    print(np.round(df.memory_usage(deep = True).sum() / 1024**2, 2))



sales = pd.read_csv(data_path + 'sales_train.csv')

sales = downcast_dtypes(sales)



items = pd.read_csv(data_path + 'items.csv')

items = downcast_dtypes(items)



item_categories = pd.read_csv(data_path + 'item_categories.csv')

item_categories = downcast_dtypes(item_categories)



shops = pd.read_csv(data_path + 'shops.csv')

shops = downcast_dtypes(shops)



test = pd.read_csv(data_path + 'test.csv')

test = downcast_dtypes(test)



display_df_info(sales, 'Sales')



display_df_info(items, 'items')



display_df_info(item_categories, 'item Categories')



display_df_info(shops, 'shops')



display_df_info(test, 'Test set')



sales_sampled = sales.sample(n = 10000)

sns.pairplot(sales_sampled[['date_block_num', 'shop_id', 'item_id', 'item_price', 'item_cnt_day']], diag_kind = 'kde')

plt.show()



del sales_sampled

gc.collect()



sns.boxplot(x = sales['item_price'])

plt.show()



sales.loc[:, 'item_price'] = sales.loc[:, 'item_price'].clip(-1, 10**5)

sale_with_negative_price = sales[sales['item_price'] < 0]

sale_with_negative_price



sale = sales[(sales.shop_id == 32) & (sales.item_id == 2973) & (sales.date_block_num == 4) & (sales.item_price > 0)]

median = sale.item_price.median()

sales.loc[sales.item_price < 0, 'item_price'] = median



del sale 

del median

del sale_with_negative_price

gc.collect()



sns.boxplot(sales['item_cnt_day'])

plt.show()



sales_temp = sales[sales['item_cnt_day'] > 500]

print('Sold item outliers')

items[items['item_id'].isin(sales_temp['item_id'].values)].merge(sales_temp[['item_id', 'item_cnt_day', 'date_block_num']], on = 'item_id')



del sales_temp

gc.collect()



print('Number of duplicates:', len(sales[sales.duplicated()]))



sales = sales.drop_duplicates(keep = 'first')

print('Number of duplicates:', len(sales[sales.duplicated()]))



start = time.time()

sales.date = sales.date.apply(lambda x: datetime.datetime.strptime(x, '%d.%m.%Y'))

print('First sale took place: ', sales.date.min())

print('Last sale took place: ', sales.date.max())

print('It tooks: ', round(time.time() - start), 'seconds')





start = time.time()

pairs_trans = create_record_for_features(sales, ['shop_id', 'item_id'], 'item_cnt_day', 'date_block_num', aggfunc = np.count_nonzero, fill = 0)

print('It tooks: ', round(time.time() - start), 'seconds')



for month in range(1, 12):

    pairs_temp = pairs_trans[['shop_id', 'item_id']][pairs_trans.loc[:,'0': str(month)].sum(axis = 1) == 0]

    print('From month: 0 until ', month,', ', np.round(100 * len(pairs_temp) / len(pairs_trans), 2), '% of the item/shop pairs have made no sales')



for month in range(21, 33):

    pairs_temp = pairs_trans[['shop_id', 'item_id']][pairs_trans.loc[:,str(month): '33'].sum(axis = 1) == 0]

    print('From month: ', month, ' until month: 33, ', np.round(100 * len(pairs_temp) / len(pairs_trans), 2), '% of the item/shop pairs have made no sales')



del pairs_trans

del pairs_temp

gc.collect()



start = time.time()

items_trans = create_record_for_features(sales, ['item_id'], 'item_cnt_day', 'date_block_num', aggfunc = np.count_nonzero, fill = 0)

print('It tooks: ', round(time.time() - start), 'seconds')





for month in range(1, 12):

    items_temp = items_trans['item_id'][items_trans.loc[:,'0': str(month)].sum(axis = 1) == 0]

    print('From month: 0 until ', month,', ', np.round(100 * len(items_temp) / len(items_trans), 2), '% of the items have made no sales')





for month in range(21, 33):

    items_temp = items_trans['item_id'][items_trans.loc[:,str(month): '33'].sum(axis = 1) == 0]

    print('From month: ', month, ' until month: 33, ', np.round(100 * len(items_temp) / len(items_trans), 2), '% of the items have made no sales')



del items_trans

del items_temp

gc.collect()



start = time.time()

shops_trans = create_record_for_features(sales, ['shop_id'], 'item_cnt_day', 'date_block_num', aggfunc = np.count_nonzero, fill = 0)

print('It tooks: ', round(time.time() - start), 'seconds')





for month in range(1, 12):

    shops_temp = shops_trans['shop_id'][shops_trans.loc[:,'0': str(month)].sum(axis = 1) == 0]

    print('From month: 0 until ', month,', ', np.round(100 * len(shops_temp) / len(shops_trans), 2), '% of the shops have made no sales')



for month in range(21, 33):

    shops_temp = shops_trans['shop_id'][shops_trans.loc[:, str(month): '33'].sum(axis = 1) == 0]

    print('From month: ', month, ' until month: 33, ', np.round(100 * len(shops_temp) / len(shops_trans), 2), '% of the shops have made no sales')



del shops_trans

del shops_temp

gc.collect()

import pandas as pd

multiple_choice_responses = pd.read_csv("../input/kaggle-survey-2019/multiple_choice_responses.csv")

other_text_responses = pd.read_csv("../input/kaggle-survey-2019/other_text_responses.csv")

questions_only = pd.read_csv("../input/kaggle-survey-2019/questions_only.csv")

survey_schema = pd.read_csv("../input/kaggle-survey-2019/survey_schema.csv")