# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from IPython.display import display

from sklearn import metrics

from sklearn.model_selection import train_test_split

# import numpy as np

# import pandas as pd

pd.set_option('display.float_format', lambda x: '%.5f' % x)
# データタイプ指定

types_dict_train = {'train_id': 'int64', 'item_condition_id': 'int8', 'price': 'float64', 'shipping': 'int8'}

types_dict_test = {'train_id': 'int64', 'item_condition_id': 'int8', 'shipping': 'int8'}



# tsvファイルからPandas DataFrameへ読み込み

train = pd.read_csv('../input/train.tsv', delimiter='\t', low_memory=True, dtype=types_dict_train)

test = pd.read_csv('../input/test.tsv', delimiter='\t', low_memory=True, dtype=types_dict_test)
# trainとtestのidカラム名を変更する

train = train.rename(columns={'train_id': 'id'})

test = test.rename(columns={'test_id': 'id'})



# 両方のセットへ is_train のColumnを追加

# 1 = trainのデータ, 0 = testデータ

train['is_train'] = 1

test['is_train'] = 0



# trainのprice以外のデータをtestと連結

train_test_combine = pd.concat([train.drop(['price'], axis=1), test], axis=0)



# 念のためデータの中身を表示

print(train_test_combine.head())



# train_test_combineの文字列のデータタイプをcategoryに変換

train_test_combine.category_name = train_test_combine.category_name.astype('category')

train_test_combine.item_description = train_test_combine.item_description.astype('category')

train_test_combine.name = train_test_combine.name.astype('category')

train_test_combine.brand_name = train_test_combine.brand_name.astype('category')



# combined data の文字列を.cat.codesで数値へ変換

train_test_combine.category_name = train_test_combine.category_name.cat.codes

train_test_combine.item_description = train_test_combine.item_description.cat.codes

train_test_combine.name = train_test_combine.name.cat.codes

train_test_combine.brand_name = train_test_combine.brand_name.cat.codes



# データの中身とデータ形式を表示して確認

print(train_test_combine.head())

print(train_test_combine.dtypes)
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

df_train['price'] = df_train['price'].apply(lambda x: np.log(x) if x > 0 else x)

 

# df_trainを表示して確認

df_train.head()
# x = price 以外のすべての値，y = priceできりわける

x_train, y_train = df_train.drop(['id', 'price'], axis=1), df_train.price



# モデルの作成

m = RandomForestRegressor(n_jobs=-1, min_samples_leaf=5, n_estimators=200)

m.fit(x_train, y_train)



# show score

print(m.score(x_train, y_train))
# 作成したランダムフォレストのモデル「m」に「df_test」を入れて予測する

x_test = df_test.drop(['id'], axis=1)

preds = m.predict(x_test)

 

# 予測値 predsをnp.exp()で処理

np.exp(preds)

 

# Numpy配列からpandasシリーズへ変換

preds = pd.Series(np.exp(preds))

 

# テストデータのIDと予測値を連結

submit = pd.concat([df_test.id, preds], axis=1)

 

# カラム名をメルカリの提出指定の名前をつける

submit.columns = ['test_id', 'price']

 

# 提出ファイルとしてCSVへ書き出し

submit.to_csv('submit_rf_base.csv', index=False)