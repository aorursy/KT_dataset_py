# まずは必要なライブラリをimportしよう。

import numpy as np

import pandas as pd

pd.options.display.max_columns = 200
# csvファイルを読み込みます。

df_train = pd.read_csv('../input/machine-learning-homework/train.csv', index_col=0)

df_test = pd.read_csv('../input/machine-learning-homework/test.csv', index_col=0)
# データを特徴量とターゲットに分割しておきます。

y_train = df_train.TradePrice

X_train = df_train.drop(['TradePrice'], axis=1)



X_test = df_test.copy()
# 区のリストを作る。ここでは面倒なので杉並区と目黒区のみ。

ward_list = ['Suminami Ward', 'Meguro Ward']
# PrefectureがTokyoかつMunicipalityがward_listにあるものを抽出してみると以下の通り。

X_train[(X_train.Prefecture=='Tokyo')&(X_train.Municipality.isin(ward_list))]
# 該当箇所に1を立てる。

X_train['Tokyo_Ward'] = 0

X_train.loc[(X_train.Prefecture=='Tokyo')&(X_train.Municipality.isin(ward_list)), 'Tokyo_Ward'] = 1
X_train