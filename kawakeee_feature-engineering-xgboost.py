import pandas as pd

import numpy as np
# データの読み込み

train = pd.read_csv('../input/competitive-data-science-predict-future-sales/sales_train.csv')

test = pd.read_csv('../input/competitive-data-science-predict-future-sales/test.csv')
# データサイズの確認

print(train.shape)

print(test.shape)
# 2013年1月から2015年10月までの日次履歴データ

# 各カラムの意味

# date:日付(表示形式:日.月.年っぽい)

# date_block_num:年月ごとの連番

# shop_id:店ID

# item_id:アイテムID

# item_price:商品価格

# item_cnt_day:その日に販売された製品の数

train.head()
train.columns = ['date', 'date_block_num', 'shop_id', 'item_id', 'item_price', 'item_cnt_day']

train
train[train.date_block_num==0].sort_values(['shop_id','item_id'], ascending=True)
train['date_block_num'] += 1

train['date_block_num'] 
# 2015年11月のショップIDとアイテムIDの売上を予測していく

# 各カラムの意味

# ID:インデックス

# shop_id:店ID

# item_id:アイテムID

test.head()
# shop.csvの確認

shops = pd.read_csv('../input/competitive-data-science-predict-future-sales/shops.csv');

print(shops.shape)

# 各カラムの意味

# shop_name:店名

# shop_id:店ID

shops.head()
# item.csvの確認

items= pd.read_csv('../input/competitive-data-science-predict-future-sales/items.csv');

print(items.shape)

# 各カラムの意味

# item_name:商品名

# item_id:商品ID

# item_category_id:商品カテゴリID

items.head()
# item_categories.csvの確認

cats= pd.read_csv('../input/competitive-data-science-predict-future-sales/item_categories.csv');

print(cats.shape)

# 各カラムの意味

# item_category_name:商品カテゴリ名

# item_category_id:商品カテゴリID

cats.head()
# sample_submission.csvの確認

sample_submission= pd.read_csv('../input/competitive-data-science-predict-future-sales/sample_submission.csv');

print(sample_submission.shape)

# 各カラムの意味

# ID:インデックス(test.csvに紐づく)

# item_cnt_month:その月に販売された製品の数

sample_submission.head()
import matplotlib.pyplot as plt

import seaborn as sns



# trainデータにて、「製品の個数」を箱ひげ図で確認する

fig,ax = plt.subplots(2,1,figsize=(10,4))

# 尺度の調整

plt.xlim(-300, 3000)

# 箱ひげ図を描画

ax[0].boxplot((train.item_cnt_day) , labels=['train.item_cnt_day'], vert=False)



# trainデータにて、「商品の価格」を箱ひげ図で確認する

plt.xlim(-1000, 350000)

ax[1].boxplot((train.item_price) , labels=['train.item_price'], vert=False)

plt.show()
# 外れ値の除外

train = train[train.item_price<100000]

train = train[train.item_cnt_day<1001]
# 0以下の値

train[train.item_price<0]
# 同じ年月/店ID/商品IDの中央値を median に代入

median = train[(train.date_block_num==4)&(train.shop_id==32)&(train.item_id==2973)&(train.item_price>0)].item_price.median()

# median を0以下の値に代入

train.loc[train.item_price<0, 'item_price'] = median

# 代入されたため、train.item_priceにて0以下の値が存在しないことを確認

train[train.item_price<0]
# 店情報を確認(全部で60店しかありません)

shops
# 重複していた店名のIDを統一させます。(train/test両方で処理しておきます)

# Якутск Орджоникидзе, 56

train.loc[train.shop_id == 0, 'shop_id'] = 57

test.loc[test.shop_id == 0, 'shop_id'] = 57

# Якутск ТЦ "Центральный"

train.loc[train.shop_id == 1, 'shop_id'] = 58

test.loc[test.shop_id == 1, 'shop_id'] = 58

# Жуковский ул. Чкалова 39м²

train.loc[train.shop_id == 10, 'shop_id'] = 11

test.loc[test.shop_id == 10, 'shop_id'] = 11
from sklearn.preprocessing import LabelEncoder

# Сергиев Посад = セルギエフ・ポサドがスペースで空いてしまっているので、このスペースを埋めます。

shops.loc[shops.shop_name == 'Сергиев Посад ТЦ "7Я"', 'shop_name'] = 'СергиевПосад ТЦ "7Я"'

# shop_nameの先頭を抽出してshopに新たな列[city(都市)]を追加します

shops['city'] = shops['shop_name'].str.split(' ').map(lambda x: x[0])

# 都市名の先頭に[!]がゴミ(タイポらしい)として入ってしまっているので、これを修正する

shops.loc[shops.city == '!Якутск', 'city'] = 'Якутск'

# LabelEncoderを使って数値化します。

shops['city_code'] = LabelEncoder().fit_transform(shops['city'])

# shopsの構造を['shop_id', 'city_code']に設定する

shops = shops[['shop_id','city_code']]
# 前処理後こんな感じになります

shops.head()
cats
# '-'でカテゴリ名を分割します

cats['split'] = cats['item_category_name'].str.split('-')

# typeには-で分割した先頭の値を代入します

cats['type'] = cats['split'].map(lambda x: x[0].strip())

# 中にはサブタイプを持たないデータもあるので、その場合はsubtypeにタイプを代入します

cats['subtype'] = cats['split'].map(lambda x: x[1].strip() if len(x) > 1 else x[0].strip())

# LabelEncoderを使って数値化します。

cats['type_code'] = LabelEncoder().fit_transform(cats['type'])

cats['subtype_code'] = LabelEncoder().fit_transform(cats['subtype'])

# shopsの構造を['item_category_id', 'type_code', 'subtype_code']に設定する

cats = cats[['item_category_id','type_code', 'subtype_code']]
# 前処理後こんな感じになります

cats.head()
items
items.drop(['item_name'], axis=1, inplace=True)
items.head()
# test(item_id) - (test(item_id) 積集合 train(item_id)) = trainに存在しないtestの商品IDの数

print(len(list(set(test.item_id) - set(test.item_id).intersection(set(train.item_id)))))

# testの商品IDの数(重複は除く)

print(len(list(set(test.item_id))))

# testの総数

print(len(test))
# train.date_block_numにて、列(Series型)ごとにユニークな値を確認(年月の組み合わせは33パターンと確認できる)

train.date_block_num.unique()
import time

# 複数のリストの直積（デカルト積）を生成するためのライブラリ

from itertools import product

ts = time.time()



# 訓練データに存在する、(年月番号,店ID,商品ID)の全組み合わせを列挙した行列を生成していく

# 最終的にmatrixを学習モデルの訓練データとする

matrix = []

for i in range(34):

    # 変数salesにdate_block_num=iの行列(表)データを代入する

    sales = train[train.date_block_num==i]

    # trainデータに存在する、(年月番号,店ID,商品ID)の全組み合わせを列挙した行列を追加していく

    matrix.append(np.array(list(product([i], sales.shop_id.unique(), sales.item_id.unique())), dtype='int16'))



# 列名を改めて設定してmatrixを更新

cols = ['date_block_num','shop_id','item_id']

matrix = pd.DataFrame(np.vstack(matrix), columns=cols)

# .astype(~~~):各特徴量を~~~でキャスト

matrix['date_block_num'] = matrix['date_block_num'].astype(np.int8)

matrix['shop_id'] = matrix['shop_id'].astype(np.int8)

matrix['item_id'] = matrix['item_id'].astype(np.int16)

# colsをソート対象とする、inplace=Trueでオブジェクトをそのまま更新

matrix.sort_values(cols,inplace=True)

time.time() - ts

# 20.83844780921936
matrix.shape
# trainデータに revenue(その日の収支合計) を追加します

train['revenue'] = train['item_price'] *  train['item_cnt_day']

train.head()
# trainデータにて、'date_block_num','shop_id','item_id'でGROUP化したDataFrameGroupByオブジェクトに対して、'item_cnt_day'を集計します

group = train.groupby(['date_block_num','shop_id','item_id']).agg({'item_cnt_day': ['sum']})

# 列名の更新

group.columns = ['item_cnt_month']

# DataFrameGroupBy -> DataFrame に変換

group.reset_index(inplace=True)

group.head()
# DataFrame同士でcolsを条件に左結合する

matrix = pd.merge(matrix, group, on=cols, how='left')

# item_cnt_monthの前処理

matrix['item_cnt_month'] = (matrix['item_cnt_month']

                                .fillna(0) # 0で穴埋めする

                                .clip(0,20) # 最小値0/最大値20に収める(なぜこの値？)

                                .astype(np.float16)) # 型のキャスト

matrix.shape
 # 2015年11月のデータのためdate_block_num = 34として列を追加してあげましょう

test['date_block_num'] = 34

# 型のキャスト

test['date_block_num'] = test['date_block_num'].astype(np.int8)

test['shop_id'] = test['shop_id'].astype(np.int8)

test['item_id'] = test['item_id'].astype(np.int16)

test.head()
test.shape
ts = time.time()

# matrixにtestを連結させる

matrix = pd.concat([matrix, test], ignore_index=True, sort=False, keys=cols)

# NaNを0に変換

matrix.fillna(0, inplace=True) # 34 month

time.time() - ts

matrix[matrix.date_block_num==34]
ts = time.time()

# Shop/Cat/Itemの特徴量をmatrixに追加する

matrix = pd.merge(matrix, shops, on=['shop_id'], how='left')

matrix = pd.merge(matrix, items, on=['item_id'], how='left')

matrix = pd.merge(matrix, cats, on=['item_category_id'], how='left')

# 型のキャスト

matrix['city_code'] = matrix['city_code'].astype(np.int8)

matrix['item_category_id'] = matrix['item_category_id'].astype(np.int8)

matrix['type_code'] = matrix['type_code'].astype(np.int8)

matrix['subtype_code'] = matrix['subtype_code'].astype(np.int8)

time.time() - ts

matrix
def lag_feature(df, lags, col):

    tmp = df[['date_block_num','shop_id','item_id',col]]

    for i in lags:

        shifted = tmp.copy()

        # 列名の更新

        shifted.columns = ['date_block_num','shop_id','item_id', col+'_lag_'+str(i)]

        shifted['date_block_num'] += i

        df = pd.merge(df, shifted, on=['date_block_num','shop_id','item_id'], how='left')

    return df
matrix
ts = time.time()

# [1,2,3,6,12]に格納された値のヶ月前のitem_cnt_monthを特量量として追加する

matrix = lag_feature(matrix, [1,2,3,6,12], 'item_cnt_month')

time.time() - ts
matrix
# import gc

# import pickle

# matrix.to_pickle('data.pkl')

# # del matrix

# # del cache

# # del group

# # del items

# # del shops

# # del cats

# # del train

# # leave test for submission

# gc.collect();
# data = pd.read_pickle('data.pkl')
# matrix.columns
# data = data[matrix.columns]
# # 訓練用データ

# X_train = data[data.date_block_num < 33].drop(['item_cnt_month'], axis=1)

# Y_train = data[data.date_block_num < 33]['item_cnt_month']

# # バリデーション用データ

# X_valid = data[data.date_block_num == 33].drop(['item_cnt_month'], axis=1)

# Y_valid = data[data.date_block_num == 33]['item_cnt_month']

# # 

# X_test = data[data.date_block_num == 34].drop(['item_cnt_month'], axis=1)
# del data

# gc.collect();
# from xgboost import XGBRegressor



# ts = time.time()

# model = XGBRegressor(

#     max_depth=8,

#     n_estimators=1000,

#     min_child_weight=300, 

#     colsample_bytree=0.8, 

#     subsample=0.8, 

#     eta=0.3,    

#     seed=42)



# model.fit(

#     X_train, 

#     Y_train, 

#     eval_metric="rmse", 

#     eval_set=[(X_train, Y_train), (X_valid, Y_valid)], 

#     verbose=True, 

#     early_stopping_rounds = 10)



# time.time() - ts
# Y_pred = model.predict(X_valid).clip(0, 20)

# Y_test = model.predict(X_test).clip(0, 20)



# submission = pd.DataFrame({

#     "ID": test.index, 

#     "item_cnt_month": Y_test

# })

# submission.to_csv('xgb_submission.csv', index=False)



# # save predictions for an ensemble

# pickle.dump(Y_pred, open('xgb_train.pickle', 'wb'))

# pickle.dump(Y_test, open('xgb_test.pickle', 'wb'))
# plot_features(model, (10,14))