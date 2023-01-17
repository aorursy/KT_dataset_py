import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import seaborn as sns

import matplotlib.pyplot as plt

from scipy.stats import skew

from scipy.special import boxcox1p

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import StandardScaler, RobustScaler

from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import Lasso, LassoCV



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv') #訓練データ

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv') #テストデータ

houces = pd.concat([train,test],sort=False)





train.describe()

test.describe()
houces.info()
pd.set_option('display.max_rows', 100)

houces.isnull().sum()[houces.isnull().sum()>0].sort_values(ascending=False)
train.info()
# 欠損値の対応

missing = houces.isnull().sum()

missing = missing[missing > 0]

missing.sort_values(inplace=True)

missing.plot.bar()
# 文字列系の欠損値があるものをチェック

houces.select_dtypes(include='object').isnull().sum()[houces.select_dtypes(include='object').isnull().sum()>0]
# 最頻値で補完

for col in ('MSZoning','Exterior1st','Exterior2nd','KitchenQual','SaleType','Functional'):

    train[col]=train[col].fillna(train[col].mode()[0])

    test[col]=test[col].fillna(test[col].mode()[0])



# Noneで補完 →「ない」ことがデータである可能性が高い。(プールがない。とか)

for col in ('Alley','Utilities','MasVnrType','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1',

            'BsmtFinType2','Electrical','FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond',

           'PoolQC','Fence','MiscFeature'):

    train[col]=train[col].fillna('None')

    test[col]=test[col].fillna('None')
# 数値系の欠損値があるデータをチェック

houces.select_dtypes(include=['int','float']).isnull().sum()[houces.select_dtypes(include=['int','float']).isnull().sum()>0]
# LotFrontage：家の前の道路の幅 →平均で埋めておく

# MasVnrArea : 石材料を使っている面積(そもそもあるかどうかの変数もある)　→０で良い

# BsmtFinSF1：タイプ1完成した平方フィート　→数ないし、０

# BsmtFinSF2：タイプ2完成した平方フィート →０

# BsmtUnfSF：地下室の未完成の平方フィート　→０

# TotalBsmtSF：地下室の総平方フィート →０

# BsmtFullBath：地下室のフルバスルーム　→０

# BsmtHalfBath：地下半分のバスルーム　→0

# GarageYrBlt：ガレージが建設された年 →　平均

# GarageCars：車の容量におけるガレージのサイズ　→　０

# GarageArea：平方フィート単位のガレージのサイズ　→0



# 平均で補完

for col in ('LotFrontage','GarageCars'):

    train[col]=train[col].fillna(train[col].mean())

    test[col]=test[col].fillna(train[col].mean())



# 0で補完

for col in ('MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath','GarageYrBlt','GarageCars','GarageArea'):

    train[col]=train[col].fillna(0)

    test[col]=test[col].fillna(0)



houces = pd.concat([train,test],sort=False)

houces.isnull().sum()
plt.figure(figsize=[30,15])

sns.heatmap(train.corr(), annot=True)
# これだと、数値データしか反映されていないので、

# Label Encoderを使う

# (文字形式のデータだと、この後何かと計算できないので、適当に同じ文字列は数字に変える、という処理を行う)

# (例：['apple','banana','grape'] → [0,1,2])



from sklearn.preprocessing import LabelEncoder



for i in range(train.shape[1]):

    if train.iloc[:,i].dtypes == object:

        lbl = LabelEncoder()

        lbl.fit(list(train.iloc[:,i].values) + list(test.iloc[:,i].values))

        train.iloc[:,i] = lbl.transform(list(train.iloc[:,i].values))

        test.iloc[:,i] = lbl.transform(list(test.iloc[:,i].values))

plt.figure(figsize=[30,15])



sns.heatmap(train.corr(), annot=False)
# 多すぎて意味がわからんので、SalesPrice（目的変数との相関係数が高いものトップ2０だけでだす）

corrmat = train.corr() #変数同士の相関係数を出してくれる。ヒートマップ に表示されている数字と一致するはず

top_corr_features = corrmat.nlargest(21,'SalePrice')['SalePrice'].index

plt.subplots(figsize=(12,10))

sns.heatmap(train[top_corr_features].corr(), annot=True)
sns.set()

columns = ['SalePrice','OverallQual','TotalBsmtSF','GrLivArea','GarageCars','FullBath','YearBuilt','YearRemodAdd']

sns.pairplot(train[columns],size = 2 ,kind ='scatter',diag_kind='kde')

plt.show()
# 他の方法

# ランダムフォレストのfeature_importanceというメソッドを使えば、どの変数が予測に大切そうかチェックできる



y_train = train['SalePrice']

X_train = train.drop(['Id','SalePrice'], axis=1)

X_test = test.drop('Id', axis=1)



# feature importance using random forest

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=80, max_features='auto')

rf.fit(X_train, y_train)



ranking = np.argsort(-rf.feature_importances_)

f, ax = plt.subplots(figsize=(11, 9))

sns.barplot(x=rf.feature_importances_[ranking], y=X_train.columns.values[ranking], orient='h')

ax.set_xlabel("feature importance")

plt.tight_layout()

plt.show()
