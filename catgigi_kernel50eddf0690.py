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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
%matplotlib inline

# データをデータフレームとして読み込む
items = pd.read_csv('../input/competitive-data-science-predict-future-sales/items.csv')
item_categories = pd.read_csv('../input/competitive-data-science-predict-future-sales/item_categories.csv')
shops = pd.read_csv('../input/competitive-data-science-predict-future-sales/shops.csv')
sales_train = pd.read_csv('../input/competitive-data-science-predict-future-sales/sales_train.csv')
test = pd.read_csv('../input/competitive-data-science-predict-future-sales/test.csv')
item_categories.head()
# ' - 'で文字列分割
item_categories['big_category_name'] = item_categories['item_category_name'].map(lambda x: x.split(' - ')[0])
# 集約具合を確認
item_categories['big_category_name'].value_counts()
# Чистые носители (штучные),Чистые носители (шпиль)は一緒？
item_categories.loc[
    item_categories['big_category_name']=='Чистые носители (штучные)',
    'big_categry'
] = 'Чистые носители (шпиль)'
# 再度集約具合を確認
item_categories['big_category_name'].value_counts()
shops.head()
shops['city_name'] = shops['shop_name'].map(lambda x: x.split(' ')[0])
# 集約具合を確認
shops['city_name'].value_counts()
shops.loc[shops['city_name']=='!Якутск','city_name'] = 'Якутск'
# 再度集約具合を確認
shops['city_name'].value_counts()
# 日次売り上げ額
sales_train['date_sales'] = sales_train['item_cnt_day'] * sales_train['item_price']
# 月次shop_id*item_id別売上点数
mon_shop_item_cnt = sales_train[
    ['date_block_num','shop_id','item_id','item_cnt_day']
].groupby(
    ['date_block_num','shop_id','item_id'],
    as_index=False
).sum().rename(columns={'item_cnt_day':'mon_shop_item_cnt'})
# 月次shop_id*item_id別売上金額
mon_shop_item_sales = sales_train[
    ['date_block_num','shop_id','item_id','date_sales']
].groupby(
    ['date_block_num','shop_id','item_id'],
    as_index=False
).sum().rename(columns={'date_sales':'mon_shop_item_sales'})
# 学習データセットをフルに拡張
# 34月*shop_id*item_id
train_full_comb = pd.DataFrame()
for i in range(35):
    mid = test[['shop_id','item_id']]
    mid['date_block_num'] = i
    train_full_comb = pd.concat([train_full_comb,mid],axis=0)
# 月次売上商品数
train = pd.merge(
    train_full_comb,
    mon_shop_item_cnt,
    on=['date_block_num','shop_id','item_id'],
    how='left'
)
# 月次売上金額
train = pd.merge(
    train,
    mon_shop_item_sales,
    on=['date_block_num','shop_id','item_id'],
    how='left'
)
# 学習データにマスタをマージ
# item_idのjoin
train = pd.merge(
    train,
    items[['item_id','item_category_id']],
    on='item_id',
    how='left'
)
# item_categry_idのjoin
train = pd.merge(
    train,
    item_categories[['item_category_id','big_category_name']],
    on='item_category_id',
    how='left'
)
# shop_idのjoin
train = pd.merge(
    train,
    shops[['shop_id','city_name']],
    on='shop_id',
    how='left'
)
plt_df = train.groupby(
    ['date_block_num'],
    as_index=False
).sum()
plt.figure(figsize=(20, 10))
sns.lineplot(x='date_block_num',y='mon_shop_item_cnt',data=plt_df)
plt.title('Montly item counts')
plt_df = train.groupby(
    ['date_block_num','big_category_name'],
    as_index=False
).sum()
plt.figure(figsize=(20, 10))
sns.lineplot(x='date_block_num',y='mon_shop_item_cnt',data=plt_df,hue='big_category_name')
plt.title('Montly item counts by big category')
plt_df = train.groupby(
    ['date_block_num','city_name'],
    as_index=False
).sum()
plt.figure(figsize=(20, 10))
sns.lineplot(x='date_block_num',y='mon_shop_item_cnt',data=plt_df,hue='city_name')
plt.title('Montly item counts by city_name')
# 月次売上数をクリップ
train['mon_shop_item_cnt'] = train['mon_shop_item_cnt'].clip(0,20)
# ラグ生成対象のカラム
lag_col_list = ['mon_shop_item_cnt','mon_shop_item_sales']
# ラグリスト(1ヶ月前、3ヶ月前、6ヶ月前、9ヶ月前、12ヶ月前)
lag_num_list = [1,3,6,9,12]

# shop_id*item_id*date_block_numでソート
train = train.sort_values(
    ['shop_id', 'item_id','date_block_num'],
    ascending=[True, True,True]
).reset_index(drop=True)

# ラグ特徴量の生成
for lag_col in lag_col_list:
    for lag in lag_num_list:
        set_col_name =  lag_col + '_' +  str(lag)
        df_lag = train[['shop_id', 'item_id','date_block_num',lag_col]].sort_values(
            ['shop_id', 'item_id','date_block_num'],
            ascending=[True, True,True]
        ).reset_index(drop=True).shift(lag).rename(columns={lag_col: set_col_name})
        train = pd.concat([train, df_lag[set_col_name]], axis=1)
# 欠損を0埋め
train = train.fillna(0)
# ラグで最大12ヶ月前の売上数を使用するため
train_ = train[(train['date_block_num']<=33) & (train['date_block_num']>=12)].reset_index(drop=True)
test_ = train[train['date_block_num']==34].reset_index(drop=True)

# モデルに入力する特徴量とターゲット変数に分割
train_y = train_['mon_shop_item_cnt']
train_X = train_.drop(columns=['date_block_num','mon_shop_item_cnt', 'mon_shop_item_sales'])
test_X = test_.drop(columns=['date_block_num','mon_shop_item_cnt', 'mon_shop_item_sales'])
from sklearn.preprocessing import LabelEncoder
obj_col_list = ['big_category_name','city_name']
for obj_col in obj_col_list:
    le = LabelEncoder()
    train_X[obj_col] = pd.DataFrame({obj_col:le.fit_transform(train_X[obj_col])})
    test_X[obj_col] = pd.DataFrame({obj_col:le.fit_transform(test_X[obj_col])})
from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor()
rfr.fit(train_X,train_y)
plt.figure(figsize=(20, 10))
sns.barplot(
    x=rfr.feature_importances_,
    y=train_X.columns.values
)
plt.title('Importance of features')
test_y = rfr.predict(test_X)
test_X['item_cnt_month'] = test_y
submission = pd.merge(
    test,
    test_X[['shop_id','item_id','item_cnt_month']],
    on=['shop_id','item_id'],
    how='left'
)
# 提出ファイル作成
submission[['ID','item_cnt_month']].to_csv('../output/submission.csv', index=False)