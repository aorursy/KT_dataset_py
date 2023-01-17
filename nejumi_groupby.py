# まずは必要なライブラリをimportしよう。

import numpy as np

import pandas as pd

pd.options.display.max_columns = 200



import lightgbm as lgb

from lightgbm import LGBMRegressor



from sklearn.preprocessing import LabelEncoder

from sklearn.experimental import enable_hist_gradient_boosting

from sklearn.ensemble import HistGradientBoostingRegressor

from sklearn.model_selection import GroupKFold

from sklearn.metrics import mean_squared_error, mean_squared_log_error



from tqdm import tqdm_notebook as tqdm
# csvファイルを読み込みます。

df_train = pd.read_csv('../input/machine-learning-homework/train.csv', index_col=0)

df_test = pd.read_csv('../input/machine-learning-homework/test.csv', index_col=0)
df_train.head()
# "MinTimeToNearestStation"が欠損している行を確認しておく

df_train[df_train.MinTimeToNearestStation.isnull()==True]
# わかりやすくするため、上記の欠損行のインデックスを控えておく

null_ix = df_train[df_train.MinTimeToNearestStation.isnull()==True].index
# 'Municipality'ごとに'MinTimeToNearestStation'の中央値を集計しておく

summary = df_train.groupby(['Municipality'])['MinTimeToNearestStation'].median()

summary
# 欠損部分に集計結果を代入する

df_train.loc[null_ix, 'MinTimeToNearestStation'] = df_train.Municipality.map(summary)
# "MinTimeToNearestStation"が欠損していた行が所属する"Municipality"のごとの中央値で埋められていることを確認する

df_train.loc[null_ix]