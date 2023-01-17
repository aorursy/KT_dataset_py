# まずは必要なライブラリをimportしよう。

import numpy as np

import pandas as pd

pd.options.display.max_columns = 200



from category_encoders import OneHotEncoder
# csvファイルを読み込みます。

df_train = pd.read_csv('../input/machine-learning-homework/train.csv', index_col=0)

df_test = pd.read_csv('../input/machine-learning-homework/test.csv', index_col=0)
# 先頭5行を確認してみよう。

df_train.head()
# データを特徴量とターゲットに分割しておきます。

y_train = df_train.TradePrice

X_train = df_train.drop(['TradePrice'], axis=1)



X_test = df_test.copy()
# unique valuesを確認しておく

X_train.Type.value_counts()
X_test.Type.value_counts()
# "Type"をOne-Hot Encodingしてみる

ohe = OneHotEncoder()

encoded_train = ohe.fit_transform(X_train.Type)

encoded_test = ohe.transform(X_test.Type)
# 結果確認

encoded_train.head()
encoded_test.head()
# 横方向(axis=1)に結合(concat)する。

X_train = pd.concat([X_train, encoded_train], axis=1)

X_test = pd.concat([X_test, encoded_test], axis=1)
# 元特徴量は削除しても良いし、さらに別途活用しても良い。

X_train.drop(['Type'], axis=1, inplace=True)

X_test.drop(['Type'], axis=1, inplace=True)
# 右端にconcatされていることを確認する。

X_train.head()
X_test.head()