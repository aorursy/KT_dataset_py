import lightgbm as lgbm

import xgboost as xgb

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

import pandas as pd

import pandas_profiling as pdp

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import datetime

%matplotlib inline

sns.set()



pd.set_option('display.max_columns', None)



# %% [markdown]

# ## Load data



# home = '/kaggle/input/competitive-data-science-predict-future-sales/'

home = './competitive-data-science-predict-future-sales/'

home = '/kaggle/input/competitive-data-science-predict-future-sales/'



items = pd.read_csv(home + 'items.csv')

categories = pd.read_csv(home + 'item_categories.csv')

shops = pd.read_csv(home + 'shops.csv')

train = pd.read_csv(home + 'sales_train.csv')

test = pd.read_csv(home + 'test.csv')
# pdp.ProfileReport(train)
# pdp.ProfileReport(test)
plt.figure(figsize=(10, 3))

sns.boxplot(x='item_cnt_day', data=train)

plt.show()

plt.figure(figsize=(10, 3))

sns.boxplot(x='item_price', data=train)

plt.show()
train.loc[train['item_cnt_day'] >= 1000,

          'item_cnt_day'] = train['item_cnt_day'].median()

train.loc[train['item_price'] >= 100000,

          'item_price'] = train['item_price'].median()
train.sort_values('item_cnt_day').head()
train.sort_values('item_price').head()

train.loc[train['item_price'] < 0, 'item_price'] = train['item_price'].median()



train['sales_day'] = train['item_price'] * train['item_cnt_day']
shops.loc[shops['shop_name'] == 'Сергиев Посад ТЦ "7Я"',

          'shop_name'] = 'СергиевПосад ТЦ "7Я"'

shops['city'] = shops['shop_name'].map(lambda x: x.split(' ')[0])

shops.loc[shops['city'] == '!Якутск', 'city'] = 'Якутск'

shops['city_code'] = LabelEncoder().fit_transform(shops['city'])

shops = shops[['shop_id', 'city_code']]
categories['splitted'] = categories['item_category_name'].map(

    lambda x: x.split(' - '))

categories['cats'] = categories['splitted'].map(lambda x: x[0].strip())

categories['subcats'] = categories['splitted'].map(

    lambda x: x[1].strip() if len(x) > 1 else x[0].strip())

categories.loc[categories['cats'] ==

               'Чистые носители (штучные)', 'cats'] = 'Чистые носители'

categories.loc[categories['cats'] ==

               'Чистые носители (шпиль)', 'cats'] = 'Чистые носители'

categories['cats_code'] = LabelEncoder().fit_transform(categories['cats'])

categories['subcats_code'] = LabelEncoder(

).fit_transform(categories['subcats'])

categories = categories[['item_category_id', 'cats_code', 'subcats_code']]
items = items.drop(['item_name'], axis=1)
dataset = train[

    ['date_block_num', 'shop_id', 'item_id', 'item_cnt_day', 'sales_day']

].groupby(

    ['date_block_num', 'shop_id', 'item_id'], as_index=False

).sum().rename(columns={'item_cnt_day': 'item_cnt_month', 'sales_day': 'sales_month'})

dataset['item_cnt_month'] = dataset['item_cnt_month'].clip(0, 20)
plt.figure(figsize=(10, 7))

plt.title('Monthly item counts')

sns.lineplot(x='date_block_num', y='item_cnt_month', data=dataset)

plt.show()

plt.figure(figsize=(10, 7))

plt.title('Monthly sales')

sns.lineplot(x='date_block_num', y='sales_month', data=dataset)

plt.show()
test['date_block_num'] = 34
all_df = pd.concat([dataset, test], keys=['date_block_num',

                                          'shop_id', 'item_id'], ignore_index=True, sort=False)

all_df = all_df.fillna(0)
all_df = pd.merge(all_df, shops, on=['shop_id'], how='left')

all_df = pd.merge(all_df, items, on=['item_id'], how='left')

all_df = pd.merge(all_df, categories, on=['item_category_id'], how='left')
def create_lag(df, lags, col):

    tmp_df = df[['date_block_num', 'shop_id', 'item_id', col]]

    for lag in lags:

        col_name = col + '_lag' + str(lag)

        copied = tmp_df.copy()

        copied.columns = ['date_block_num', 'shop_id', 'item_id', col_name]

        copied['date_block_num'] += lag

        df = pd.merge(df, copied, on=[

                      'date_block_num', 'shop_id', 'item_id'], how='left')



    return df
all_df = create_lag(all_df, [1, 2, 3, 6, 9, 12], 'item_cnt_month')

all_df = all_df.fillna(0)
def create_mean(df, col):

    col_name = 'item_cnt_mean_by_' + col

    tmp_df = df.groupby(['date_block_num', col]).agg({

        'item_cnt_month': ['mean']})

    tmp_df.columns = tmp_df.columns.droplevel(0)

    tmp_df = tmp_df.rename(columns={'mean': col_name})

    df = pd.merge(df, tmp_df, on=['date_block_num', col], how='left')



    return df
cols = ['shop_id', 'item_id', 'city_code', 'cats_code', 'subcats_code']



for col in cols:

    all_df = create_mean(all_df, col)
all_df.head()
all_df = all_df.drop(['ID'], axis=1)

all_df = all_df[all_df['date_block_num'] >= 12].reset_index(drop=True)

all_df.info()
X = all_df[all_df['date_block_num']

                 <= 33].drop(['item_cnt_month'], axis=1)

y = all_df[all_df['date_block_num'] <= 33]['item_cnt_month']

X_test = all_df[all_df['date_block_num'] == 34].drop(

    ['item_cnt_month'], axis=1)



X_train, X_valid, y_train, y_valid = train_test_split(

                                                        X,

                                                        y, 

                                                        test_size=0.3,

                                                        random_state=42

                                                     )
model_lgbm = lgbm.LGBMRegressor()

model_lgbm.fit(

            X_train, 

            y_train,

            eval_set=[(X_valid, y_valid)],

            eval_metric='rmse',

            early_stopping_rounds=10

         )
model_xgb = xgb.XGBRegressor()



model_xgb.fit(

            X_train, 

            y_train, 

            eval_set=[(X_valid, y_valid)],

            eval_metric='rmse',

            early_stopping_rounds=10

         )
model_rfr = RandomForestRegressor(oob_score=True)

model_rfr.fit(X_train, y_train)

score = model_rfr.oob_score_

print(score)
# y_pred = model.predict(X_test, num_iteration=model.best_iteration_).clip(0, 20)

y_pred = model_lgbm.predict(X_test)
sub = pd.DataFrame(test['ID'])

sub['item_cnt_month'] = y_pred



sub.to_csv('submission.csv', index=False)