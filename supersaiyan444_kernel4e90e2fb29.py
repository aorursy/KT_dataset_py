import numpy as np
import pandas as pd


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# train.csv と test.csv を読み込み
train = pd.read_csv(filepath_or_buffer="../input/house-prices-advanced-regression-techniques/train.csv",
                    encoding="ms932", sep=",")
test = pd.read_csv(filepath_or_buffer="../input/house-prices-advanced-regression-techniques/test.csv",
                   encoding="ms932", sep=",")
# train と test を表示
# print(train) # コメントアウト外すと表示
# print(test)  # コメントアウト外すと表示
# train と test のデータの型を表示
# print(train.dtypes) # コメントアウト外すと表示
# print(test.dtypes)  # コメントアウト外すと表示
# データの種類を全て数値に変換
from sklearn.preprocessing import LabelEncoder
for i in range(train.shape[1]):
    if train.iloc[:,i].dtypes == object:
        lbl = LabelEncoder()
        lbl.fit(list(train.iloc[:,i].values) + list(test.iloc[:,i].values))
        train.iloc[:,i] = lbl.transform(list(train.iloc[:,i].values))
        test.iloc[:,i] = lbl.transform(list(test.iloc[:,i].values))
# 欠損値(NaN)の確認
train_nan = train.isnull().sum()
train_nan = train_nan[train_nan > 0]
test_nan = test.isnull().sum()
test_nan = test_nan[test_nan > 0]
# print(train_nan) # コメントアウト外すと表示
# print(test_nan)  # コメントアウト外すと表示
# 欠損値の削除
train_del = train.drop(['LotFrontage','MasVnrArea','GarageYrBlt'], axis=1)
test_del  = test.drop(['LotFrontage','MasVnrArea','GarageYrBlt'], axis=1)
# 欠損値を中央値で穴埋め
test_fill = test_del.fillna(test_del.median())
# train と test からデータの削除
train_x = train_del.drop(['Id','SalePrice'], axis=1)
test_x = test_fill.drop('Id', axis=1)
# train からデータの抜き出し
train_y = train_del['SalePrice']
test_id = test_fill['Id']
# 線形回帰
from sklearn.linear_model import LinearRegression
slr = LinearRegression()
slr.fit(train_x,train_y)
test_y = slr.predict(test_x)
# print(test_y) # 線形回帰後の数値を表示
# 提出ファイルの作成
sub = pd.DataFrame({
    "Id": test_id,
    "SalePrice": test_y
})
sub.to_csv('sub.csv',index=False)