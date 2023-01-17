

import pandas as pd

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from IPython.display import display

from sklearn import metrics

from sklearn.model_selection import train_test_split

pd.set_option('display.float_format', lambda x:'%.5f' % x)

import numpy as np

path = '../input/mercari-price-suggestion-challenge/'
col_names = [ 'c{0:02d}'.format(i) for i in range(10) ]

# データタイプを指定

types_dict_train = {'train_id':'int64', 'item_condition_id':'int8', 'price':'float64', 'shipping':'int8'}

types_dict_test = {'test_id':'int64', 'item_condition_id':'int8', 'shipping':'int8'}

 

# tsvファイルからPandas DataFrameへ読み込

train = pd.read_csv(path+'train.tsv.7z', delimiter='\t', low_memory=True, dtype=types_dict_train)

test = pd.read_csv(path+'test.tsv.7z', delimiter='\t', low_memory=True, dtype=types_dict_test)


# trainとtestのデータフレームの冒頭5行を表示させる

train.head()

test.head()

 

# trainとtestのサイズを確認

train.shape, test.shape

 


def display_all(df):

    with pd.option_context("display.max_rows", 1000):

        with pd.option_context("display.max_columns", 1000):

            display(df)

 

# trainの基本統計量を表示

display_all(train.describe(include='all').transpose())

 
# trainのカテゴリ名、商品説明、投稿タイトル、ブランド名のデータタイプを「category」へ変換する

train.category_name = train.category_name.astype('category')

train.item_description = train.item_description.astype('category')

train.name = train.name.astype('category')

train.brand_name = train.brand_name.astype('category')

 

# testのカテゴリ名、商品説明、投稿タイトル、ブランド名のデータタイプを「category」へ変換する

test.category_name = test.category_name.astype('category')

test.item_description = test.item_description.astype('category')

test.name = test.name.astype('category')

test.brand_name = test.brand_name.astype('category')

 

# dtypesで念のためデータ形式を確認しましょう

train.dtypes, test.dtypes


# trainの中のユニークな値を確認する

train.apply(lambda x: x.nunique())

 

# testの中のユニークな値を確認する

test.apply(lambda x: x.nunique())

 

 


# trainの欠損データの個数と%を確認

train.isnull().sum(),train.isnull().sum()/train.shape[0]

 

# testの欠損データの個数と%を確認

test.isnull().sum(),test.isnull().sum()/test.shape[0]

 

 


# trainとtestのidカラム名を変更する

train = train.rename(columns = {'train_id':'id'})

test = test.rename(columns = {'test_id':'id'})

 

# 両方のセットへ「is_train」のカラムを追加

# 1 = trainのデータ、0 = testデータ

train['is_train'] = 1

test['is_train'] = 0

 

# trainのprice(価格）以外のデータをtestと連結

train_test_combine = pd.concat([train.drop(['price'], axis=1),test],axis=0)

 

# 念のためデータの中身を表示させましょう

train_test_combine.head()

 

 


# train_test_combineの文字列のデータタイプを「category」へ変換

train_test_combine.category_name = train_test_combine.category_name.astype('category')

train_test_combine.item_description = train_test_combine.item_description.astype('category')

train_test_combine.name = train_test_combine.name.astype('category')

train_test_combine.brand_name = train_test_combine.brand_name.astype('category')

 

# combinedDataの文字列を「.cat.codes」で数値へ変換する

train_test_combine.name = train_test_combine.name.cat.codes

train_test_combine.category_name = train_test_combine.category_name.cat.codes

train_test_combine.brand_name = train_test_combine.brand_name.cat.codes

train_test_combine.item_description = train_test_combine.item_description.cat.codes

 

# データの中身とデータ形式を表示して確認しましょう

train_test_combine.head()

train_test_combine.dtypes

 


# 「is_train」のフラグでcombineからtestとtrainへ切り分ける

df_test = train_test_combine.loc[train_test_combine['is_train'] == 0]

df_train = train_test_combine.loc[train_test_combine['is_train'] == 1]

 

# 「is_train」をtrainとtestのデータフレームから落とす

df_test = df_test.drop(['is_train'], axis=1)

df_train = df_train.drop(['is_train'], axis=1)

 

# サイズの確認をしておきましょう

df_test.shape, df_train.shape


# df_trainへprice（価格）を戻す

df_train['price'] = train.price

 

# price（価格）をlog関数で処理

df_train['price'] = df_train['price'].apply(lambda x: np.log(x) if x>0 else x)

 

# df_trainを表示して確認

df_train.head()


# x ＝ price以外の全ての値、y = price（ターゲット）で切り分ける

x_train, y_train = df_train.drop(['price'], axis=1), df_train.price

 

# モデルの作成

m = RandomForestRegressor(n_jobs=-1, min_samples_leaf=5, n_estimators=200, verbose=10)



m.fit(x_train, y_train)

 

# スコアを表示

m.score(x_train, y_train)

 


# 作成したランダムフォレストのモデル「m」に「df_test」を入れて予測する

preds = m.predict(df_test)

 

# 予測値 predsをnp.exp()で処理

np.exp(preds)

 

# Numpy配列からpandasシリーズへ変換

preds = pd.Series(np.exp(preds))

 

# テストデータのIDと予測値を連結

submit = pd.concat([df_test.id, preds], axis=1)

 

# カラム名をメルカリの提出指定の名前をつける

submit.columns = ['test_id', 'price']

 

# 提出ファイルとしてCSVへ書き出し

submit.to_csv('output/submit_rf_base.csv', index=False)