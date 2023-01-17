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
from matplotlib import pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LinearRegression

from scipy.spatial.distance import cdist

%matplotlib inline
# データの読み込み

train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv') # 訓練データ

test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')   # テストデータ

# 学習データとテストデータのマージ

train['WhatIsData'] = 'Train'

test['WhatIsData'] = 'Test'

test['SalePrice'] = 9999999999

alldata = pd.concat([train,test],axis=0).reset_index(drop=True)
#学習データの変数を確認

train.columns
# 学習データの欠損状況

train.isnull().sum()[train.isnull().sum()>0].sort_values(ascending=False)
# テストデータの欠損状況

test.isnull().sum()[test.isnull().sum()>0].sort_values(ascending=False)
# 欠損を含むカラムのデータ型を確認

na_col_list = alldata.isnull().sum()[alldata.isnull().sum()>0].index.tolist() # 欠損を含むカラムをリスト化

alldata[na_col_list].dtypes.sort_values() #データ型
# データ型に応じて欠損値を補完する

# floatの場合は0

# objectの場合は'NA'

na_float_cols = alldata[na_col_list].dtypes[alldata[na_col_list].dtypes=='float64'].index.tolist() #float64

na_obj_cols = alldata[na_col_list].dtypes[alldata[na_col_list].dtypes=='object'].index.tolist() #object

# float64型で欠損している場合は0を代入

for na_float_col in na_float_cols:

    alldata.loc[alldata[na_float_col].isnull(),na_float_col] = 0.0

# object型で欠損している場合は'NA'を代入

for na_obj_col in na_obj_cols:

    alldata.loc[alldata[na_obj_col].isnull(),na_obj_col] = 'NA'
# マージデータの欠損状況

alldata.isnull().sum()[alldata.isnull().sum()>0].sort_values(ascending=False)
# 相関係数の値を確認（ヒートマップの作成）

k = 11

df = train

corrmat = df.corr()

cols = corrmat.nlargest(k, "SalePrice")["SalePrice"].index

cm = np.corrcoef(df[cols].values.T)

fig, ax = plt.subplots(figsize=(12, 10))

sns.set(font_scale=1.2)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt=".2f", annot_kws={"size": 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()

fig.savefig("figure.png")
# 目的変数の分布を確認

sns.distplot(train['SalePrice'])
sns.distplot(np.log(train['SalePrice']))
# マージデータを学習データとテストデータに分割

train_ = alldata[alldata['WhatIsData']=='Train'].drop(['WhatIsData','Id'], axis=1).reset_index(drop=True)

test_ = alldata[alldata['WhatIsData']=='Test'].drop(['WhatIsData','SalePrice'], axis=1).reset_index(drop=True)



# 一部の学習データ内の分割(一部)

train_y = np.log(train_['SalePrice'])

# テストデータ内の分割(一部)

test_id = test_['Id']
# 予測モデルの構築

# 説明変数の指定(3:相関係数0.7以上，7:相関係数0.6以上，11:相関係数0.5以上)

num = [3, 7, 11]

cnt = 0

for i in num:

    #データ内の分割

    train_x = train_[cols[1:i]]

    test_data = test_[cols[1:i]]

    # 線形回帰モデルの呼び出し

    model = LinearRegression()

    # 学習用データと検証用データに分割

    x_train, x_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.3, random_state=0)

    # モデルの訓練

    model.fit(x_train, y_train)

    # それぞれのデータのスコアを算出

    train_rmse = np.sqrt(mean_squared_error(y_train, model.predict(x_train)))

    test_rmse = np.sqrt(mean_squared_error(y_test, model.predict(x_test)))

    # ベストスコアの更新

    if cnt == 0:

        best_score = test_rmse

        best_num = i

    elif best_score > test_rmse:

        best_score = test_rmse

        best_num = i

    else:

        pass

    print('num : ' + str(i))

    print('score is : ' +str(test_rmse))

    cnt = cnt + 1

    

print('test num is : ' + str(best_num))

print('test score is : ' + str(best_score))
# 提出用データ生成

train_x = train_[cols[1:best_num]]

test_data = test_[cols[1:best_num]]

model = LinearRegression()

model.fit(train_x, train_y)

test_SalePrice = pd.DataFrame(np.exp(model.predict(test_data)),columns=['SalePrice'])

test_Id = pd.DataFrame(test_id,columns=['Id'])

pd.concat([test_Id, test_SalePrice],axis=1).to_csv('/kaggle/working/output.csv',index=False)
