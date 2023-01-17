# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from IPython.display import display
from sklearn import metrics
from sklearn.model_selection import train_test_split
pd.set_option("display.float_format", lambda x:"%.5f" % x)
import numpy as np
# データタイプの指定
types_dict_train = {"train_id": "int64", "item_condition_id": "int8", "price": "float64", "shipping": "int8"}
types_dict_test = {"test_id": "int64", "item_conditon_id": "int8", "shipping": "int8"}

# tsvファイルからPandas DataFrameへ読み込み
train = pd.read_csv("../input/train.tsv", delimiter="\t", low_memory=True, dtype=types_dict_train)
test = pd.read_csv("../input/test.tsv", delimiter="\t", low_memory=True, dtype=types_dict_test)
# trainのデータフレームの冒頭5行を表示
train.head()
# trainとtestのサイズを確認
train.shape, test.shape
# 基本統計量の確認
# Pandasの行と列を表示するオプションを最大にして表示「display_all」
# 行と列を転置する「transpose()」
def display_all(df):
    with pd.option_context("display.max_rows", 1000):
        with pd.option_context("display.max_columns", 1000):
            display(df)

# trainの基本統計量
display_all(train.describe(include='all').transpose())
# trainのカテゴリ名、商品説明、投稿タイトル、ブランド名のデータタイプを「category」へ変換
train.category_name = train.category_name.astype("category")
train.item_description = train.item_description.astype("category")
train.name = train.name.astype("category")
train.brand_name = train.brand_name.astype("category")

# testのカテゴリ名、商品説明、投稿タイトル、ブランド名のデータタイプを「category」へ変換
test.category_name = test.category_name.astype("category")
test.item_description = test.item_description.astype("category")
test.name = test.name.astype("category")
test.brand_name = test.brand_name.astype("category")

# dtypesでデータ形式の確認
train.dtypes, test.dtypes
# trainの中のユニークな値を確認
train.apply(lambda x: x.nunique())
# trainの欠損データの個数と%を確認
# 「shape[0]」は行の数。「shape[1]」にすると列の数
train.isnull().sum(), train.isnull().sum()/train.shape[0]
# testの欠損データの個数と％を確認
# 「shape[0]」は行の数。「shape[1]」にすると列の数
test.isnull().sum(), test.isnull().sum()/test.shape[0]
# trainとtestのidカラム名をidに変更して、idを揃える
train = train.rename(columns = {"train_id": "id"})
test = test.rename(columns = {"test_id": "id"})

# 両方のセットへ「is_train」のカラムを追加し、trainとtestの判別をできるようにしておく。
# trainのis_trainカラムに1を挿入、testのis_trainカラムに0を挿入
train["is_train"] = 1
test["is_train"] = 0

# trainのprice（価格）以外のデータをtestと連結
train_test_combine = pd.concat([train.drop(["price"], axis=1),test],axis=0)

# 連結後のデータ確認
train_test_combine.head()
# train_test_combineの文字列のデータタイプを「category」へ変換
train_test_combine.name = train_test_combine.name.astype("category")
train_test_combine.category_name = train_test_combine.category_name.astype("category")
train_test_combine.brand_name = train_test_combine.brand_name.astype("category")
train_test_combine.item_description = train_test_combine.item_description.astype("category")

# ランダムフォレストのモデルを作るために、文字列のデータを「.cat.codes」を使って数値に変換する
train_test_combine.name = train_test_combine.name.cat.codes
train_test_combine.category_name = train_test_combine.category_name.cat.codes
train_test_combine.brand_name = train_test_combine.brand_name.cat.codes
train_test_combine.item_description = train_test_combine.item_description.cat.codes

# データの中身を確認
train_test_combine.head()
# 「is_train」のカラムで、train_test_combineからtrainとtestに再び切り分ける
# 「loc」は行or列のラベルを指定して値を取得する
df_train = train_test_combine.loc[train_test_combine["is_train"] == 1]
df_test = train_test_combine.loc[train_test_combine["is_train"] == 0]

# trainとtestどちらに属するかのフラグは不要になったので、「is_train」カラムをtrainとtestのデータフレームから消す
df_train = df_train.drop(["is_train"], axis=1)
df_test = df_test.drop(["is_train"], axis=1)

# サイズを確認
df_train.shape, df_test.shape
# df_trainへprice（価格）を戻す
df_train["price"] = train.price

# price（価格）をlog関数で処理
# 「apply」はDataFrameの各行or各列に関数を適用する
# 「np.log()」は、対数の計算。ex:2を底とする8の対数は3。2*2*2
df_train["price"] = df_train["price"].apply(lambda x: np.log(x) if x>0 else x)

# df_trainを表示
df_train.head()
# x = price以外で全ての値、y = price（ターゲット）で切り分ける
x_train, y_train = df_train.drop(["price"], axis=1), df_train.price

# モデルの作成
m = RandomForestRegressor(n_jobs=-1, min_samples_leaf=5, n_estimators=200)
m.fit(x_train, y_train)

# スコア表示
m.score(x_train, y_train)
# 作成したランダムフォレストのモデル「m」に「df_test」を入れて予測をする
preds = m.predict(df_test)

# 予測値predsを指数関数「np.exp()」で処理
np.exp(preds)

# Numpy配列からpandasのSeriesへ変換
preds = pd.Series(np.exp(preds))

# テストデータのIDと予測値を連結
submit = pd.concat([df_test.id, preds], axis=1)

# カラム名をメルカリの提出指定の名前にする
submit.columns = ["test_id", "price"]

# 提出ファイルとしてcsv書き出し
submit.to_csv("submit_rf_base.csv", index=False)
