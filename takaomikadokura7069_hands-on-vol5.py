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
import warnings
#ワーニングを抑止
warnings.filterwarnings('ignore')
%matplotlib inline
# 小数点2桁で表示(指数表記しないように)
pd.options.display.float_format = '{:.2f}'.format

# データをデータフレームとして読み込む
items = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/items.csv')
item_categories = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/item_categories.csv')
shops = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/shops.csv')
sales_train = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv')
test = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/test.csv')

#日次売上データの表示
sales_train.head()
sales_train.shape
fig = plt.figure(figsize=(18,9))
plt.subplots_adjust(hspace=.5)
sales_train['shop_id'].value_counts(normalize=True).plot(kind='bar')
plt.title('Shop ID Values in the Training Set (Normalized)')

#sales_trainの月別件数を表示
fig = plt.figure(figsize=(18,9))
plt.subplots_adjust(hspace=.5)
itm_cnt = sales_train['date_block_num'].value_counts(normalize=True).plot.bar()
plt.title('Month (date_block_num) Values in the Training Set (Normalized)')

#商品カテゴリマスタの表示
item_categories.head()
# ' - 'で文字列分割し、big_category_nameとitem_category_nameに分ける。
item_categories['big_category_name'] = item_categories['item_category_name'].map(lambda x: x.split(' - ')[0])
# 集約具合を確認
item_categories['big_category_name'].value_counts()

# 表記揺れがあるので統一
# Чистые носители (штучные) と
# Чистые носители (шпиль)を同一とする
item_categories.loc[
    item_categories['big_category_name']=='Чистые носители (штучные)'
] = 'Чистые носители (шпиль)'

# 再度集約具合を確認
item_categories['big_category_name'].value_counts()

item_categories.head()
#テストデータの表示
test.head()
shops.head()

# ' 'で文字列分割し、city_nameとshop_nameに分ける
shops['city_name'] = shops['shop_name'].map(lambda x: x.split(' ')[0])
# 集約具合を確認
shops['city_name'].value_counts()

# 表記揺れがあるので統一
# !Якутск と
# Якутскを同一とする
shops.loc[shops['city_name']=='!Якутск','city_name'] = 'Якутск'
# 再度集約具合を確認
shops['city_name'].value_counts()

shops.head()

#商品数と商品価格の散布図を作成
plt.figure(figsize=(20, 10))
plt.scatter(sales_train["item_cnt_day"],sales_train["item_price"])
plt.xlabel("sales_train.item_cnt_day")
plt.ylabel("sales_train.item_price")

# 外れ値の除外
# 商品価格>100000 および 商品数>1001 の外れ値を訓練データから削除
sales_train = sales_train[sales_train.item_price<100000]
sales_train = sales_train[sales_train.item_cnt_day<1001]

#商品数と商品価格の散布図を作成
plt.figure(figsize=(20, 10))
plt.scatter(sales_train["item_cnt_day"],sales_train["item_price"])
plt.xlabel("sales_train.item_cnt_day")
plt.ylabel("sales_train.item_price")

# 日次売上額を作成(商品数×商品価格)
sales_train['date_sales'] = sales_train['item_cnt_day'] * sales_train['item_price']
sales_train.head()

# 月次店別商品別売上点数
mon_shop_item_cnt = sales_train[
    ['date_block_num','shop_id','item_id','item_cnt_day']
].groupby(
    ['date_block_num','shop_id','item_id'],
    as_index=False
).sum().rename(columns={'item_cnt_day':'mon_shop_item_cnt'})
mon_shop_item_cnt.head()

# 月次店別商品別売上金額
mon_shop_item_sales = sales_train[
    ['date_block_num','shop_id','item_id','date_sales']
].groupby(
    ['date_block_num','shop_id','item_id'],
    as_index=False
).sum().rename(columns={'date_sales':'mon_shop_item_sales'})
mon_shop_item_sales.head()

# 学習データセットをテストデータに合わせ拡張する
# 34月*shop_id*item_id
train_full_comb = pd.DataFrame()
for i in range(35):
    mid = test[['shop_id','item_id']]
    mid['date_block_num'] = i
    train_full_comb = pd.concat([train_full_comb,mid],axis=0)
train_full_comb.head()

#  月次売上商品数をレフトアウタージョイン
train = pd.merge(
    train_full_comb,
    mon_shop_item_cnt,
    on=['date_block_num','shop_id','item_id'],
    how='left'
)

# 月次売上金額をレフトアウタージョイン
train = pd.merge(
    train,
    mon_shop_item_sales,
    on=['date_block_num','shop_id','item_id'],
    how='left'
)

# 商品マスタをレフトアウタージョイン
train = pd.merge(
    train,
    items[['item_id','item_category_id']],
    on='item_id',
    how='left'
)
# 商品カテゴリマスタをレフトアウタージョイン
train = pd.merge(
    train,
    item_categories[['item_category_id','big_category_name']],
    on='item_category_id',
    how='left'
)
# 店マスタをレフトアウタージョイン
train = pd.merge(
    train,
    shops[['shop_id','city_name']],
    on='shop_id',
    how='left'
)

train.head()

plt_df = train.groupby('date_block_num',as_index=False)['mon_shop_item_cnt'].sum()
plt.figure(figsize=(20, 10))
sns.lineplot(x='date_block_num',y='mon_shop_item_cnt',data=plt_df)
plt.title('Montly item counts')

plt_df = train.groupby('date_block_num',as_index=False)['mon_shop_item_cnt'].sum()
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
    ['date_block_num','big_category_name'],
    as_index=False
).sum()
plt.figure(figsize=(20, 10))
sns.lineplot(x='date_block_num',y='mon_shop_item_cnt',data=plt_df,hue='big_category_name')
plt.title('Montly item counts by big category')

plt_df = train.groupby(
    ['date_block_num','big_category_name'],
    as_index=False
).sum()
plt.figure(figsize=(20, 10))
sns.lineplot(x='date_block_num',y='mon_shop_item_cnt',data=plt_df,hue='big_category_name')
plt.title('Montly item counts by big category')

# ラグ生成対象のカラム
lag_col_list = ['mon_shop_item_cnt','mon_shop_item_sales']
# ラグリスト(1ヶ月前、2ヶ月前、3ヶ月前)
lag_num_list = [1,2,3]

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
train.head() 

# 欠損を0埋め
train = train.fillna(0)
train.isnull().sum()

from sklearn.preprocessing import LabelEncoder

obj_col_list = ['big_category_name','city_name']
for obj_col in obj_col_list:
    le = LabelEncoder()
    train[obj_col] = pd.DataFrame({obj_col:le.fit_transform(train[obj_col])})
train.head()

# ラグで最大3ヶ月前の売上数を使用するため
train_ = train[(train['date_block_num']<=32) & (train['date_block_num']>=3)].reset_index(drop=True)
test_ = train[train['date_block_num']==33].reset_index(drop=True)

# モデルに入力する特徴量とターゲット変数に分割
y_train = train_['mon_shop_item_cnt']
X_train = train_.drop(columns=['date_block_num','mon_shop_item_cnt', 'mon_shop_item_sales'])
y_test = test_['mon_shop_item_cnt']
X_test = test_.drop(columns=['date_block_num','mon_shop_item_cnt', 'mon_shop_item_sales'])

#件数、項目数を表示
print(y_train.shape)
print(X_train.shape)
print(y_test.shape)
print(X_test.shape)

#LightGBMライブラリ
import lightgbm as lgb

lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test)

#ハイパーパラメータ
params = {'metric': {'rmse'},
          'max_depth' : 9}

#LightGBMの実行
gbm = lgb.train(params,
                lgb_train,
                valid_sets=(lgb_train, lgb_eval),
                num_boost_round=10000,
                early_stopping_rounds=100,
                verbose_eval=50)
y_pred = gbm.predict(X_test)

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

#RMSE(平均平方二乗誤差)
print(np.sqrt(mean_squared_error(y_test, y_pred)))
#MAE(平均絶対誤差)
print(mean_absolute_error(y_test, y_pred))
# 決定係数
print(r2_score(y_test, y_pred))

#特徴量の重要度
lgb.plot_importance(gbm, height=0.5, figsize=(8,16))
