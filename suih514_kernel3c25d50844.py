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
import pandas as pd

import pandas_profiling

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

import seaborn as sns

import numpy as np

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Ridge
train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
train["SaleCondition"].unique()
train
#欠損値の表示

train.isnull().sum()[train.isnull().sum()>0].sort_values(ascending=False)
train_ID = train['Id']

test_ID = test['Id']



#訓練データを分割

y_train = train['SalePrice']

X_train = train.drop(['Id','SalePrice'], axis=1)

X_test = test.drop('Id', axis=1)

#データを統合

alldata = pd.concat([X_train, X_test])
na_col_list = alldata.isnull().sum()[alldata.isnull().sum()>0].index.tolist()



#欠損があるカラムのデータ型を確認

alldata[na_col_list].dtypes.sort_values()
#floatとobjectの欠損値をそれぞれ埋める

float_list = alldata[na_col_list].dtypes[alldata[na_col_list].dtypes == "float64"].index.tolist()

obj_list = alldata[na_col_list].dtypes[alldata[na_col_list].dtypes == "object"].index.tolist()



alldata[float_list] = alldata[float_list].fillna(0)

alldata[obj_list] = alldata[obj_list].fillna("None")



#欠損値が全て置換できているか確認

alldata.isnull().sum()[alldata.isnull().sum() > 0]
from sklearn.preprocessing import LabelEncoder



# データタイプがobjectの列の値をラベル化した数値に変換(LabelEncode)

for i in range(alldata.shape[1]):

    if alldata.iloc[:,i].dtypes == object:

        lbl = LabelEncoder()

        lbl.fit(list(alldata.iloc[:,i].values))

        alldata.iloc[:,i] = lbl.transform(list(alldata.iloc[:,i].values))
# Label Encodeされた後の、SaleConditionカラムにセットされている値の一覧

alldata["SaleCondition"].unique()
#家のトータルの広さ

alldata["TotalSF"] = alldata["TotalBsmtSF"] + alldata["1stFlrSF"] + alldata["2ndFlrSF"]
#Salepriceの割合

sns.distplot(y_train)

plt.show()
#対数

y_train = np.log(y_train)



sns.distplot(y_train)

plt.show()
#再度trainデータとtestデータに分割

X_train = alldata.iloc[:train.shape[0],:]

X_test = alldata.iloc[train.shape[0]:,:]



# ランダムフォレストをインポート

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=80, max_features='auto')

rf.fit(X_train, y_train)

print("Training done using Random Forest")



#重要度の高い順にソートした配列のインデックス。

ranking = np.argsort(-rf.feature_importances_)

f, ax = plt.subplots(figsize=(11, 9))

sns.barplot(x=rf.feature_importances_[ranking], y=X_train.columns.values[ranking], orient='h')

ax.set_xlabel("feature importance")

plt.tight_layout()

plt.show()
#上位35個の特徴

X_train = X_train.iloc[:,ranking[:35]]

X_test = X_test.iloc[:,ranking[:35]]
# z-scoreにて標準化

X_train = (X_train - X_train.mean()) / X_train.std()

X_test = (X_test - X_test.mean()) / X_test.std()
#SalePriceと各特徴の関係

fig = plt.figure(figsize=(12,7))

for i in np.arange(35):

    ax = fig.add_subplot(5,7,i+1)

    sns.regplot(x=X_train.iloc[:,i], y=y_train)



plt.tight_layout()

plt.show()
#TotalSFとGrLivAreaの外れ値を削除

train = X_train

train['SalePrice'] = y_train

train = train.drop(index = train[(train['TotalSF'] > 5) & (train['SalePrice'] < 12.5)].index)

train = train.drop(index = train[(train['GrLivArea'] > 5) & (train['SalePrice'] < 13)].index)

plt.figure(figsize=(20, 10))

plt.scatter(train["OverallQual"],train["SalePrice"])

plt.xlabel("OverallQual")

plt.ylabel("SalePrice")
#OverallQualの外れ値を削除

train = train.drop(train[(train['OverallQual']<-1) & (train['SalePrice']>12.3)].index)

train = train.drop(train[(train['OverallQual']<2.5) & (train['SalePrice']>13.1)].index)



plt.figure(figsize=(20, 10))

plt.scatter(train["OverallQual"],train["SalePrice"])

plt.xlabel("OverallQual")

plt.ylabel("SalePrice")
y_train = train['SalePrice']

X_train = train.drop(['SalePrice'], axis=1)
#線形回帰での予測

ridge = Ridge(alpha=0.001)

ridge.fit(X_train,y_train)



print('傾き：{0}'.format(ridge.coef_[0]))

print('y切片: {0}'.format(ridge.intercept_))
# 学習済みのモデルから予測した結果をセット

y_test_pred = np.exp(ridge.predict(X_test))
# 学習済みのモデルから予測した結果を出力

y_test_pred
test["SalePrice"] = y_test_pred

test[["Id","SalePrice"]].head()
submission = pd.DataFrame({

    "Id": test_ID,

    "SalePrice": y_test_pred

})

submission.to_csv('submission.csv', index=False)