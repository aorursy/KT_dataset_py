# いつものライブラリ追加

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# 可視化用ライブラリ追加

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
# kaggleの入力ファイル一覧確認

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# データの読み込み

items = pd.read_csv(f'{dirname}/items.csv')

item_categories = pd.read_csv(f'{dirname}/item_categories.csv')

shops = pd.read_csv(f'{dirname}/shops.csv')

sales_train = pd.read_csv(f'{dirname}/sales_train.csv')

test = pd.read_csv(f'{dirname}/test.csv')

sample_submission = pd.read_csv(f'{dirname}/sample_submission.csv')

# 先頭5行の確認

items.head()
item_categories.head()
shops.head()
sales_train.head()
test.head()
sample_submission.head()
# item_categoriesのitem_category_nameについて確認

# uniqueを使って重複なしデータをチェック

# いくつか「-」なしのデータがある -> 確認をしたら小カテゴリがないだけらしい

# なので、-で分割して1番最初のものを大カテゴリとして採用する

item_categories['item_category_name'].unique()
# item_categories['item_category_name'] を「 - 」で分割、その1番目をcategoryとする

item_categories['category'] = item_categories['item_category_name'].map(lambda x: x.split(' - ')[0])
# 確認

# pandasの出現頻度を求めるメソッド、value_countsを使う

item_categories['category'].value_counts()
# shopのshop_nameについても同じように確認

# こっちも大体先頭データを見たのと同じようなイメージ

# なので、 で分割して1番最初のものを都市名として採用する

shops['shop_name'].unique()
# shop['shop_name'] を「 - 」で分割、その1番目をcategoryとする

shops['city'] = shops['shop_name'].map(lambda x: x.split(' ')[0])
# 確認

shops['city'].value_counts()
# sales_trainデータを整形する

# 単価と売上数から、売上金額を出す

sales_train['sales'] = sales_train['item_price'] * sales_train['item_cnt_day']
# 確認、sort_valuesで昇順にして大きさチェックも済ませる



# 指数表記で辛い

# これで小数点二桁まで表示にできる

pd.options.display.float_format = '{:.2f}'.format

# Noneがデフォルト

# pd.options.display.float_format = None

sales_train['sales'].sort_values()



# いやマイナスって何だろう、これは削った方がいいのか

# 一旦そのままにしておく
# sales_trainを再整形するため、今一度確認

# dateは日付、dd.mm.yyyy形式

# date_block_numは2013年1月を0年してそこから月ごとにまとめたものっぽい

# shop_id, item_id, date_block_numごとにitem_cnt_dayをまとめたデータと、

# shop_id, item_id, date_block_numごとにsalesをまとめたデータをつくる

sales_train
# 月ごとitem_cnt

# shop_id, item_id, date_block_numごとにitem_cnt_dayをまとめて、item_cnt_dayをリネーム

# 結合の参考 => https://qiita.com/propella/items/a9a32b878c77222630ae

mon_item_cnt = sales_train[

        ['date_block_num','shop_id','item_id','item_cnt_day']

    ].groupby(

        ['date_block_num','shop_id','item_id'],

        as_index=False

    ).sum().rename(columns={'item_cnt_day':'item_cnt'})
mon_item_cnt
# 同じように月ごとsales

# shop_id, item_id, date_block_numごとにsalesをまとめる

# 結合の参考 => https://qiita.com/propella/items/a9a32b878c77222630ae

mon_sales = sales_train[

        ['date_block_num','shop_id','item_id','sales']

    ].groupby(

        ['date_block_num','shop_id','item_id'],

        as_index=False

    ).sum()
mon_sales
# ここから訓練データの結合をする

# 34パターンのdate_block_num × shop_id × item_idのからデータを作って、

# そこにこれまでの全データを結合していく

# 予想したいものが訓練になければ0

# 逆に訓練にあるが予想する必要にないものはデータを捨てる



# 空データ作成

train = pd.DataFrame()



# date_block_num分まわす

for i in range(35):

    # テストデータから店舗idと商品idを取得する

    train_ = test[['shop_id','item_id']]

    train_['date_block_num'] = i

    # データは縦に結合

    train = pd.concat([train,train_],axis=0)
train
# 行数の確認

# 問題なければ trainにどんどんleft joinしてくっつけていく

len(test) * 35 == len(train)
# 月ごとitem_cnt

train =  pd.merge(

    train,

    mon_item_cnt,

    on=['date_block_num','shop_id','item_id'],

    how='left'

)



# 月ごとsales

train =  pd.merge(

    train,

    mon_sales,

    on=['date_block_num','shop_id','item_id'],

    how='left'

)



# itemsテーブル(item_category_idが欲しい)

train =  pd.merge(

    train,

    items[['item_id','item_category_id']],

    on=['item_id'],

    how='left'

)



# item_categoriesテーブル(categoryが欲しい)

train =  pd.merge(

    train,

    item_categories[['item_category_id','category']],

    on=['item_category_id'],

    how='left'

)





# shopssテーブル(cityが欲しい)

train =  pd.merge(

    train,

    shops[['shop_id','city']],

    on=['shop_id'],

    how='left'

)

# trainの確認

# ここまでで前処理完了

train