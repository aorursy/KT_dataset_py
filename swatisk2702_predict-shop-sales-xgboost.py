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
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
items = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/items.csv')

shops = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/shops.csv')

categories = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/item_categories.csv')

train = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv')

test  = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/test.csv')

submission = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sample_submission.csv')

train.head()
train.shape
test.head()
test.shape
submission.head()
fig = plt.figure(figsize = (18,9))

plt.subplots_adjust(hspace = 0.5)



plt.subplot2grid((3,3), (0,0), colspan = 3)

train['shop_id'].value_counts(normalize = True).plot(kind = 'bar', alpha = 0.7)

plt.title('Shop ID Values in the Training Set (Normalized)')



plt.subplot2grid((3,3), (1,0))

train['item_id'].plot(kind='hist', alpha = 0.7)

plt.title('Item ID Histogram')



plt.subplot2grid((3,3), (1,1))

train['item_price'].plot(kind='hist', alpha = 0.7, color = 'green')

plt.title('Item price Histogram')



plt.subplot2grid((3,3), (1,1))

train['item_cnt_day'].plot(kind='hist', alpha = 0.7, color = 'orange')

plt.title('Item Count day Histogram')



plt.subplot2grid((3,3), (2,0), colspan = 3)

train['date_block_num'].value_counts(normalize = True).plot(kind = 'bar', alpha = 0.7)

plt.title('Month (date_block_num) Values in the Training Set (Normalized)')



plt.show()
train['item_id'].value_counts(ascending = False)[:5]
items.loc[items.item_id == 20949]
categories.loc[categories.item_category_id == 71]
test.loc[test.item_id == 20949].head()
train['item_cnt_day'].sort_values(ascending=False)[:5]
train[train.item_cnt_day == 2169]
items[items.item_id == 11373]
train[train.item_id == 11373].head()
train[train.item_id == 11373].median()
train = train[train.item_cnt_day < 2000]
train['item_price'].sort_values(ascending = False)[:5]
train[train.item_price == 307980]
items[items.item_id == 6066]
train[train.item_id == 6066]
train = train[train.item_price < 300000]
items.head()
train['item_price'].sort_values()[:5]
train[train.item_price == -1]
train[train.item_id == 2973].head()
price_correction = train[(train.shop_id == 32) & (train.item_id == 2973) & (train.date_block_num ==4) & (train.item_price > 0)].item_price.median()

train.loc[train.item_price<0, 'item_price'] = price_correction
fig = plt.figure(figsize =(18,8) )

plt.subplots_adjust(hspace = 0.5)



plt.subplot2grid((3,3), (0,0), colspan = 3)

test['shop_id'].value_counts(normalize = True).plot(kind = 'bar', alpha =.7)

plt.title('Shop ID Values in the Test Set (Normalized)')



plt.subplot2grid((3,3), (1,0))

test['item_id'].plot(kind = 'hist', alpha = .7)

plt.title('Item ID Histogram - Test Set')



plt.show()

shops_train = train.shop_id.nunique()

shops_test = test.shop_id.nunique()

print("Shops in training set = ", shops_train)

print("Shops in test set = ", shops_test)
shops_train_list = list(train.shop_id.unique())

shops_test_list = list(test.shop_id.unique())



if set(shops_test_list).issubset(set(shops_train_list)):

    print("test shop list is subset of train shop list ")

else:

    print("test shop list is not subset of train shop list ")
shops.T
shops['city'] = shops['shop_name'].str.split(' ').map(lambda x: x[0])
from sklearn import preprocessing

LE = preprocessing.LabelEncoder()

LE.fit_transform(shops['city'])
shops['city_label'] = LE.fit_transform(shops['city'])

shops.drop(['shop_name', 'city'], axis = 1, inplace = True)

shops.head()
items_train_list = list(train.item_id.unique())

items_test_list = list(test.item_id.unique())



if set(items_test_list).issubset(set(items_train_list)):

    print("test item list is subset of train item list ")

else:

    print("test item list is not subset of train item list ")
len(set(items_test_list)-set(items_train_list))
categories.T
LE = preprocessing.LabelEncoder()

category_split = categories['item_category_name'].str.split('-')

categories['main_categories_id'] = category_split.map(lambda row: row[0].strip())

categories['main_categories_id'] = LE.fit_transform(categories['main_categories_id'])



categories['sub_category_id'] = category_split.map(lambda row: row[1].strip() if len(row)> 1 else row[0].strip())

categories['sub_category_id'] = LE.fit_transform(categories['sub_category_id'])

categories.head()
train['date'] = pd.to_datetime(train['date'], format = '%d.%m.%Y')

train.info()
from itertools import product



# Testing generation of cartesian product for the month of January in 2013



shops_in_jan = train.loc[train['date_block_num']==0, 'shop_id'].unique()

items_in_jan = train.loc[train['date_block_num']==0, 'item_id'].unique()

jan = list(product(*[shops_in_jan, items_in_jan, [0]]))

print(len(jan))
# Testing generation of cartesian product for the month of February in 2013



shops_in_feb = train.loc[train['date_block_num']==1, 'shop_id'].unique()

items_in_feb = train.loc[train['date_block_num']==1, 'item_id'].unique()

feb = list(product(*[shops_in_feb, items_in_feb, [1]]))

print(len(feb))
cartesian_test = []

cartesian_test.append(np.array(jan))

cartesian_test.append(np.array(feb))
cartesian_test = np.vstack(cartesian_test)

cartesian_test_df = pd.DataFrame(cartesian_test, columns = ['shop_id', 'item_id', 'date_block_num'])

cartesian_test_df.head()
cartesian_test_df.shape
from tqdm import tqdm_notebook



def downcast_dtypes(df):

    '''

        Changes column types in the dataframe: 

                

                `float64` type to `float32`

                `int64`   type to `int32`

    '''

    

    # Select columns to downcast

    float_cols = [c for c in df if df[c].dtype == "float64"]

    int_cols =   [c for c in df if df[c].dtype == "int64"]

    

    # Downcast

    df[float_cols] = df[float_cols].astype(np.float16)

    df[int_cols]   = df[int_cols].astype(np.int16)

    

    return df
months = train['date_block_num'].unique()



cartesian = []

for month in months:

    shops_in_month = train.loc[train['date_block_num']==month, 'shop_id'].unique()

    items_in_month = train.loc[train['date_block_num']==month, 'item_id'].unique()

    cartesian.append(np.array(list(product(*[shops_in_month, items_in_month, [month]])), dtype='int32'))

    

    

cartesian_df = pd.DataFrame(np.vstack(cartesian), columns = ['shop_id', 'item_id', 'date_block_num'], dtype=np.int32)

cartesian_df.shape
x = train.groupby(['shop_id', 'item_id', 'date_block_num'])['item_cnt_day'].sum().rename('item_cnt_month').reset_index()

x.head()
x.shape
new_train = pd.merge(cartesian_df, x, on=['shop_id', 'item_id', 'date_block_num'], how='left').fillna(0)



new_train['item_cnt_month'] = np.clip(new_train['item_cnt_month'], 0, 20)
del x

del cartesian_df

del cartesian

del cartesian_test

del cartesian_test_df

del feb

del jan

del items_test_list

del items_train_list

del train
new_train.sort_values(['date_block_num','shop_id','item_id'], inplace = True)

new_train.head()
test.insert(loc=3, column='date_block_num', value=34)

test['item_cnt_month'] = 0

test.head()
new_train = new_train.append(test.drop('ID', axis = 1))
new_train = pd.merge(new_train, shops, on=['shop_id'], how='left')

new_train.head()
new_train = pd.merge(new_train, items.drop('item_name', axis = 1), on=['item_id'], how='left')

new_train.head()
new_train = pd.merge(new_train, categories.drop('item_category_name', axis = 1), on=['item_category_id'], how='left')

new_train.head()
def generate_lag(train, months, lag_column):

    for month in months:

        # Speed up by grabbing only the useful bits

        train_shift = train[['date_block_num', 'shop_id', 'item_id', lag_column]].copy()

        train_shift.columns = ['date_block_num', 'shop_id', 'item_id', lag_column+'_lag_'+ str(month)]

        train_shift['date_block_num'] += month

        train = pd.merge(train, train_shift, on=['date_block_num', 'shop_id', 'item_id'], how='left')

    return train
del items

del categories

del shops

del test
new_train = downcast_dtypes(new_train)
import gc

gc.collect()
%%time

new_train = generate_lag(new_train, [1,2,3,4,5,6,12], 'item_cnt_month')
%%time

group = new_train.groupby(['date_block_num', 'item_id'])['item_cnt_month'].mean().rename('item_month_mean').reset_index()

new_train = pd.merge(new_train, group, on=['date_block_num', 'item_id'], how='left')

new_train = generate_lag(new_train, [1,2,3,6,12], 'item_month_mean')

new_train.drop(['item_month_mean'], axis=1, inplace=True)
%%time

group = new_train.groupby(['date_block_num', 'shop_id'])['item_cnt_month'].mean().rename('shop_month_mean').reset_index()

new_train = pd.merge(new_train, group, on=['date_block_num', 'shop_id'], how='left')

new_train = generate_lag(new_train, [1,2,3,6,12], 'shop_month_mean')

new_train.drop(['shop_month_mean'], axis=1, inplace=True)
%%time

group = new_train.groupby(['date_block_num', 'shop_id', 'item_category_id'])['item_cnt_month'].mean().rename('shop_category_month_mean').reset_index()

new_train = pd.merge(new_train, group, on=['date_block_num', 'shop_id', 'item_category_id'], how='left')

new_train = generate_lag(new_train, [1, 2], 'shop_category_month_mean')

new_train.drop(['shop_category_month_mean'], axis=1, inplace=True)
%%time

group = new_train.groupby(['date_block_num', 'main_category_id'])['item_cnt_month'].mean().rename('main_category_month_mean').reset_index()

new_train = pd.merge(new_train, group, on=['date_block_num', 'main_category_id'], how='left')



new_train = generate_lag(new_train, [1], 'main_category_month_mean')

new_train.drop(['main_category_month_mean'], axis=1, inplace=True)
%%time

group = new_train.groupby(['date_block_num', 'sub_category_id'])['item_cnt_month'].mean().rename('sub_category_month_mean').reset_index()

new_train = pd.merge(new_train, group, on=['date_block_num', 'sub_category_id'], how='left')



new_train = generate_lag(new_train, [1], 'sub_category_month_mean')

new_train.drop(['sub_category_month_mean'], axis=1, inplace=True)
new_train.tail()
new_train['month'] = new_train['date_block_num'] % 12
holiday_dict = {

    0: 6,

    1: 3,

    2: 2,

    3: 8,

    4: 3,

    5: 3,

    6: 2,

    7: 8,

    8: 4,

    9: 8,

    10: 5,

    11: 4,

}



new_train['holidays_in_month'] = new_train['month'].map(holiday_dict)
moex = {

    12: 659, 13: 640, 14: 1231,

    15: 881, 16: 764, 17: 663,

    18: 743, 19: 627, 20: 692,

    21: 736, 22: 680, 23: 1092,

    24: 657, 25: 863, 26: 720,

    27: 819, 28: 574, 29: 568,

    30: 633, 31: 658, 32: 611,

    33: 770, 34: 723,

}



new_train['moex_value'] = new_train.date_block_num.map(moex)

new_train = downcast_dtypes(new_train)
import gc

gc.collect()
import xgboost as xgb

new_train = new_train[new_train.date_block_num > 11]
def fill_na(df):

    for col in df.columns:

        if ('_lag_' in col) & (df[col].isnull().any()):

            df[col].fillna(0, inplace=True)         

    return df



new_train = fill_na(new_train)


def xgtrain():

    regressor = xgb.XGBRegressor(n_estimators = 5000,

                                 learning_rate = 0.01,

                                 max_depth = 10,

                                 subsample = 0.5,

                                 colsample_bytree = 0.5)

    

    regressor_ = regressor.fit(new_train[new_train.date_block_num < 33].drop(['item_cnt_month'], axis=1).values, 

                               new_train[new_train.date_block_num < 33]['item_cnt_month'].values, 

                               eval_metric = 'rmse', 

                               eval_set = [(new_train[new_train.date_block_num < 33].drop(['item_cnt_month'], axis=1).values, 

                                            new_train[new_train.date_block_num < 33]['item_cnt_month'].values), 

                                           (new_train[new_train.date_block_num == 33].drop(['item_cnt_month'], axis=1).values, 

                                            new_train[new_train.date_block_num == 33]['item_cnt_month'].values)

                                          ], 

                               verbose=True,

                               early_stopping_rounds = 50,

                              )

    return regressor_
%%time

regressor_ = xgtrain()
predictions = regressor_.predict(new_train[new_train.date_block_num == 34].drop(['item_cnt_month'], axis = 1).values)
from matplotlib import rcParams

rcParams['figure.figsize'] = 11.7,8.27



cols = new_train.drop('item_cnt_month', axis = 1).columns

plt.barh(cols, regressor_.feature_importances_)

plt.show()
submission['item_cnt_month'] = predictions

submission.to_csv('sales_faster_learn.csv', index=False)