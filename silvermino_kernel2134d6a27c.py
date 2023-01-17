# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#売上データでデータフレームを作成

DATA = '../input/competitive-data-science-predict-future-sales/'

sales = pd.read_csv(DATA+'sales_train.csv', parse_dates=['date'], infer_datetime_format=True, dayfirst=True)

sales.head()
#テストデータを取得

test = pd.read_csv(DATA+'test.csv')

test.head()
#売上データを月別の売上に変換

df = sales.groupby([sales.date.apply(lambda x: x.strftime('%Y-%m')),'item_id','shop_id']).sum().reset_index()

df = df[['date','item_id','shop_id','item_cnt_day']]

df = df.pivot_table(index=['item_id','shop_id'], columns='date',values='item_cnt_day',fill_value=0).reset_index()

df.head()
#月次売上データをテストデータにマージ

df_test = pd.merge(test, df, on=['item_id','shop_id'], how='left')

df_test = df_test.fillna(0)

df_test.head()
#テストデータからカテゴリデータを削除

df_test = df_test.drop(labels=['ID', 'shop_id', 'item_id'], axis=1)

df_test.head()
#学習セットを作成

#予測の対象:2015-10年の売上欄

TARGET = '2015-10'

y_train = df_test[TARGET]

X_train = df_test.drop(labels=[TARGET], axis=1)



print(y_train.shape)

print(X_train.shape)

X_train.head()
#学習集合をnumpy行列に変換

# X_train = X_train.as_matrix()

# X_train = X_train.reshape((214200, 33, 1))



# y_train = y_train.as_matrix()

# y_train = y_train.reshape(214200, 1)



print(y_train.shape)

print(X_train.shape)



# X_train[:1]
#テストデータをnumpy行列に変換してテストセットを作成

#最初の月を削除することで，学習されたLSTMが既知の時間範囲を超えて予測値を出力できるようにする

X_test = df_test.drop(labels=['2013-01'],axis=1)

# X_test = X_test.as_matrix()

# X_test = X_test.reshape((214200, 33, 1))

print(X_test.shape)
from lightgbm import LGBMRegressor
model=LGBMRegressor(

        n_estimators=200,

        learning_rate=0.03,

        num_leaves=32,

        colsample_bytree=0.9497036,

        subsample=0.8715623,

        max_depth=8,

        reg_alpha=0.04,

        reg_lambda=0.073,

        min_split_gain=0.0222415,

        min_child_weight=40)
print('Training time, it is...')

model.fit(X_train, y_train,

          

         )
#テストセットの予測値とクリップ値を指定した範囲で取得

y_pred = model.predict(X_test).clip(0., 20.)



#提出ファイルの作成

preds = pd.DataFrame(y_pred, columns=['item_cnt_month'])

preds.to_csv('submission.csv',index_label='ID')