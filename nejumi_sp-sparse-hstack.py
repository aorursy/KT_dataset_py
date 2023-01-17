# ライブラリのインポート

import numpy as np

import scipy as sp

import pandas as pd

from pandas import DataFrame



from sklearn.feature_extraction.text import TfidfVectorizer

from category_encoders import OrdinalEncoder
# データの読み込み

df_train = pd.read_csv('../input/homework-for-students2/train.csv', index_col=0)

df_test = pd.read_csv('../input/homework-for-students2/test.csv', index_col=0)
# 説明変数とターゲットに分割

y_train = df_train.loan_condition

X_train = df_train.drop(['loan_condition'], axis=1)



X_test = df_test
# テキストとそれ以外に分割

TXT_train = X_train.emp_title.copy()

TXT_test = X_test.emp_title.copy()



X_train.drop(['emp_title'], axis=1, inplace=True)

X_test.drop(['emp_title'], axis=1, inplace=True)
# とりあえずobjectになったらOrdinalEncodingする

cats = []



for col in X_train.columns:

    if X_train[col].dtype == 'object':

        cats.append(col)

        

print(cats)
encoder = OrdinalEncoder(cols=cats)



X_train[cats] = encoder.fit_transform(X_train[cats])

X_test[cats] = encoder.transform(X_test[cats])
# テキストをTFIDF

tdidf = TfidfVectorizer(max_features=1000)



TXT_train = tdidf.fit_transform(TXT_train.fillna('#'))

TXT_test = tdidf.transform(TXT_test.fillna('#'))
# TFIDFにかけたテキストをhstack

X_train = sp.sparse.hstack([X_train.values, TXT_train])

X_test = sp.sparse.hstack([X_test.values, TXT_test])
X_train
X_test
# 初めて使う際には、sparse matrixは色々戸惑う部分もあるかもしれません。例えば、CV時にindexでsliceしようとして「あれ？」と躓いたりなど。

# sparse matrixには複数の格納形式があるなどしますので、色々調べてみてください。