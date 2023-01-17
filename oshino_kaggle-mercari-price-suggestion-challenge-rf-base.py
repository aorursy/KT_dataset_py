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
from sklearn.ensemble import RandomForestRegressor
# データ読み込み
train = pd.read_table('../input/train.tsv')
test = pd.read_table('../input/test.tsv')
# 型を数値に変更
def convert_type_to_int(train, test, columns):
    train = train.rename(columns = {'train_id':'id'})
    test = test.rename(columns = {'test_id':'id'})
    train['is_train'] = 1
    test['is_train'] = 0
    df = pd.concat([train.drop(['price'], axis=1), test], axis=0)
    for column in columns:
        df[column] = df[column].astype('category')
        df[column] = df[column].cat.codes
    df_train = df.loc[df['is_train'] == 1].drop(['is_train'], axis=1).rename(columns = {'id':'train_id'})
    df_test = df.loc[df['is_train'] == 0].drop(['is_train'], axis=1).rename(columns = {'id':'test_id'})
    df_train['price'] = train.price
    return df_train, df_test
# いったんobject型は全てcategory型に変換し、数値に変換
obj_columns = ['category_name', 'brand_name', 'name', 'item_description']
train, test = convert_type_to_int(train, test, obj_columns)
# 学習のためのデータ準備
X_train = train.drop(['train_id', 'price'], axis=1) # トレーニングデータから目的変数priceとキー列をdrop
Y_train = train['price'] # トレーニングデータから目的変数を切り出し
X_test  = test.drop('test_id', axis=1).copy() # テストデータからキー列をdrop

X_train.shape, Y_train.shape, X_test.shape
# ランダムフォレストでモデルの作成
m = RandomForestRegressor(n_jobs=-1, min_samples_leaf=5, n_estimators=200)
m.fit(X_train, Y_train)
 
# スコアを表示
m.score(X_train, Y_train)
# モデルでテストデータからpriceを予測
Y_test = m.predict(X_test)
# 提出用データ作成(test_id, price列を持つcsv形式)
test_price = pd.Series(Y_test)
submission = pd.DataFrame({'test_id': test['test_id'], 'price': test_price})
submission['price'] = submission['price'].round(3) # sample通り小数点は第３位まで
submission.to_csv('submit_rf_base.csv', index=False)
