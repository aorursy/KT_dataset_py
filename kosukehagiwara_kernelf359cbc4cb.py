# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style='darkgrid')

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import ( LinearRegression, Ridge, Lasso )
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#トレーニングデータの読み込み
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
#テストデータの読み込み
test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
train.info()
test.info()
train.head()
test.head()
#欠損値をカウント、パーセンテージを出力する関数
def count_missing_rate(df):
    count = 0
    for column in df.columns:
        total = df[column].isnull().sum()#欠損値のカウント
        percent = round(total/len(df[column])*100,2)#データ数に対する欠損値の割合
        if count == 0:
            df1 = pd.DataFrame([[total,percent]], columns=['total', 'percent'], index=[column])
            count+=1
        else:#作成したカラム毎のDataFrameを結合
            df2 = pd.DataFrame([[total, percent]], columns=['total', 'percent'], index=[column])
            df1 = pd.concat([df1, df2], axis=0)
            count+=1
    return df1.query('total> 0 ')
#全てのデータを表示するコマンド
pd.set_option("display.max_rows",81)
#trainデータの欠損値の数と割合を表示
train_na = count_missing_rate(train)
train_na
#全てのデータを表示するコマンド
pd.set_option("display.max_rows",81)
#testデータの欠損値の数と割合を表示
test_na = count_missing_rate(test)
test_na
#SalePriceを追加し欠損値で埋める
test['SalePrice'] = np.nan

#testデータのIdを後で使うためここで取得しておく
test_Id = test['Id']

#trainデータとtestデータの結合
df = pd.concat([train, test], ignore_index = True, sort = False)

#結合したデータの確認
print(df.info())
pd.set_option("display.max_columns",81)
print(df.describe(include='all'))
t_na_list = test_na.index.tolist()
#float型のリスト
df_float_na = df[t_na_list].dtypes[df[t_na_list].dtypes=='float64'].index.tolist()
#object型のリスト
df_obj_na = df[t_na_list].dtypes[df[t_na_list].dtypes=='object'].index.tolist()

#float型の欠損値を０で補完する
for column in df_float_na:
    df.loc[df[column].isnull(), column]=0.0
    
#object型の欠損値をNAで補完する
for column in df_obj_na:
    df.loc[df[column].isnull(), column]='NA'
#結合していたデータをtrainデータとtestデータに分割する
#trainデータ
train = df[df['SalePrice'].notnull()]
#testデータ
test = df[df['SalePrice'].isnull()].drop('SalePrice', axis=1)
#trainデータのワンホットエンコーディング
train_objlist = train.dtypes[train.dtypes=='object'].index.tolist()
train_objdummy = pd.get_dummies(train[train_objlist])

#testデータのワンホットエンコーディング
test_objlist = test.dtypes[test.dtypes=='object'].index.tolist()
test_objdummy = pd.get_dummies(test[test_objlist])

pd.set_option("display.max_columns",81)
print(train_objdummy.describe(include='all'))
#object型以外のリストを取得し先ほどワンホットエンコーディングしたデータと結合させる

#trainデータの結合
train_numcol = train.dtypes[train.dtypes!='object'].index.tolist()
train = pd.concat([train[train_numcol], train_objdummy], axis=1)

#testデータの結合
test_numcol = test.dtypes[test.dtypes!='object'].index.tolist()
test = pd.concat([test[test_numcol], test_objdummy], axis=1)
#trainデータとtestデータの情報を確認
print(train.info())
print()
print(test.info())
#カラム名を格納する空のリストを作成
train_list = []
test_list = []

#それぞれのカラム名をリストに追加していく
for column1 in train.columns:
    train_list.append(column1)
    
for column2 in test.columns:
    test_list.append(column2)

print(len(train_list))
print(len(test_list))
print()

#trainにのみ存在するカラムリストを格納
train_only = list(set(train_list) - set(test_list))
print(len(train_only))
print(train_only)
print()


#testにのみ存在するカラムリストを格納
test_only = list(set(test_list) - set(train_list))
print(len(test_only))
print(test_only)
#trainデータに新たなカラムを追加する
for column in test_only:
    train[column] = 0

#testデータに新たなカラムを追加する
for column in train_only:
    test[column] = 0

#testデータのSalePriceを削除する
test = test.drop('SalePrice', axis=1)

print(train.info())
print()
print(test.info())
#SalePriceのヒストグラムを表示
sns.distplot(train.SalePrice)
#SalePriceを対数変換してヒストグラムで表示する
sns.distplot(np.log(train.SalePrice))
#説明変数と目的変数に分ける
train_X = train.drop('SalePrice', axis=1)
train_y = np.log(train['SalePrice'])
ss = ShuffleSplit(n_splits=10,
                 train_size=0.8,
                  test_size=0.2, 
                  random_state=0)

count = 0

alpha = 0.00099
La = Lasso(alpha=0.001, max_iter=50000)
la = []

for train_index, test_index in ss.split(train_X, train_y):
    
    count+=1
    
    X_train, X_test = train_X.iloc[train_index], train_X.iloc[test_index]
    y_train, y_test = train_y.iloc[train_index], train_y.iloc[test_index]
    
    La.fit(X_train, y_train)
    y_pred = La.predict(X_train)
    
    la.append(mean_squared_error(y_train, y_pred))
    
    print('{}回目'.format(count))
    print('Lassoのスコア ： ', mean_squared_error(y_train, y_pred))


print('Lassoのスコア平均 ： ', np.mean(la))
plt.subplots_adjust(wspace=0.4)
plt.subplot(121)
plt.scatter(np.exp(y_train),np.exp(La.predict(X_train)))
plt.subplot(122)
plt.scatter(np.exp(y_test),np.exp(La.predict(X_test)))
#提出用データの作成
solution = pd.DataFrame(np.exp(La.predict(test)), test_Id, columns=['SalePrice'])
solution.to_csv('../House Price/house_price_solution.csv', index_label = ['Id'])
solution
