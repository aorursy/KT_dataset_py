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

from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.pipeline import make_pipeline

from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import PowerTransformer

from sklearn.linear_model import (

    LinearRegression,

    Ridge,

    Lasso,

    ElasticNet

)
# データの読み込み

train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv') #訓練データ

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv') #テストデータ
print('The size of test is : ' + str(test.shape))
#物件の広さを合計した変数を作成

train["TotalSF"] = train["1stFlrSF"] + train["2ndFlrSF"] + train["TotalBsmtSF"]

test["TotalSF"] = test["1stFlrSF"] + test["2ndFlrSF"] + test["TotalBsmtSF"]



#物件の広さと物件価格の散布図を作成

plt.figure(figsize=(20, 10))

plt.scatter(train["TotalSF"],train["SalePrice"])

plt.xlabel("TotalSF")

plt.ylabel("SalePrice")
#外れ値を除外する

train = train.drop(train[(train['TotalSF']>7500) & (train['SalePrice']<300000)].index)

train = train.drop(train[(train['TotalSF']>6000) & (train['SalePrice']<500000)].index)



#物件の広さと物件価格の散布図を作成

plt.figure(figsize=(20, 10))

plt.scatter(train["TotalSF"],train["SalePrice"])

plt.xlabel("TotalSF")

plt.ylabel("SalePrice")
data = pd.concat([train["YearBuilt"],train["SalePrice"]],axis=1)



plt.figure(figsize=(20, 10))

plt.xticks(rotation='90')

sns.boxplot(x="YearBuilt",y="SalePrice",data=data)
#外れ値を除外する

train = train.drop(train[(train['YearBuilt']<2000) & (train['SalePrice']>600000)].index)



#グラフを描画する

data = pd.concat([train["YearBuilt"],train["SalePrice"]],axis=1)



plt.figure(figsize=(20, 10))

plt.xticks(rotation='90')

sns.boxplot(x="YearBuilt",y="SalePrice",data=data)
#家の材質・完成度と物件価格の散布図を作成

plt.figure(figsize=(20, 10))

plt.scatter(train["OverallQual"],train["SalePrice"])

plt.xlabel("OverallQual")

plt.ylabel("SalePrice")
#外れ値を除外する

train = train.drop(train[(train['OverallQual']<5) & (train['SalePrice']>200000)].index)

train = train.drop(train[(train['OverallQual']<10) & (train['SalePrice']>500000)].index)



#グラフを描画する

plt.figure(figsize=(20, 10))

plt.scatter(train["OverallQual"],train["SalePrice"])

plt.xlabel("OverallQual")

plt.ylabel("SalePrice")
#学習データを目的変数とそれ以外に分ける

train_x = train.drop("SalePrice",axis=1)

train_y = train["SalePrice"]



#学習データとテストデータを統合

all_data = pd.concat([train_x,test],axis=0,sort=True)



#IDのカラムは不必要なので別の変数に格納

train_ID = train['Id']

test_ID = test['Id']



all_data.drop("Id", axis = 1, inplace = True)



#それぞれのデータのサイズを確認

print("train_x: "+str(train_x.shape))

print("train_y: "+str(train_y.shape))

print("test: "+str(test.shape))

print("all_data: "+str(all_data.shape))
# 学習データの欠損状況

all_data_na = all_data.isnull().sum()[all_data.isnull().sum()>0].sort_values(ascending=False)

all_data_na
#欠損値の数をグラフ化

plt.figure(figsize=(20,10))

plt.xticks(rotation='90')

sns.barplot(x=all_data_na.index, y=all_data_na)
# 欠損を含むカラムのデータ型を確認

na_col_list = all_data.isnull().sum()[all_data.isnull().sum()>0].index.tolist() # 欠損を含むカラムをリスト化

all_data[na_col_list].dtypes.sort_values() #データ型
#隣接した道路の長さ（LotFrontage）の欠損値の補完

all_data['LotFrontage'] = all_data.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))



#欠損値が存在するかつfloat型のリストを作成

float_list = all_data[na_col_list].dtypes[all_data[na_col_list].dtypes == "float64"].index.tolist()



#欠損値が存在するかつobject型のリストを作成

obj_list = all_data[na_col_list].dtypes[all_data[na_col_list].dtypes == "object"].index.tolist()



#float型の場合は欠損値を0で置換

all_data[float_list] = all_data[float_list].fillna(0)



#object型の場合は欠損値を"None"で置換

all_data[obj_list] = all_data[obj_list].fillna("None")



#欠損値が全て置換できているか確認

all_data.isnull().sum()[all_data.isnull().sum() > 0]
# マージデータの欠損状況

all_data.isnull().sum()[all_data.isnull().sum()>0].sort_values(ascending=False)
# カテゴリ変数に変換する

all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)

all_data['YrSold'] = all_data['YrSold'].astype(str)

all_data['MoSold'] = all_data['MoSold'].astype(str)
#目的変数の対数log(x+1)をとる

train_y = np.log1p(train_y)



#分布を可視化

plt.figure(figsize=(20, 10))

sns.distplot(train_y)
#数値の説明変数のリストを作成

num_feats = all_data.dtypes[all_data.dtypes != "object" ].index



#各説明変数の歪度を計算

skewed_feats = all_data[num_feats].apply(lambda x: x.skew()).sort_values(ascending = False)



#グラフ化

plt.figure(figsize=(20,10))

plt.xticks(rotation='90')

sns.barplot(x=skewed_feats.index, y=skewed_feats)
#歪度の絶対値が0.5より大きい変数だけに絞る

skewed_feats_over = skewed_feats[abs(skewed_feats) > 0.5].index



#各変数の最小値を表示

for i in skewed_feats_over:

    print(min(all_data[i]))
#Yeo-Johnson変換

pt = PowerTransformer()

pt.fit(all_data[skewed_feats_over])



#変換後のデータで各列を置換

all_data[skewed_feats_over] = pt.transform(all_data[skewed_feats_over])



#各説明変数の歪度を計算

skewed_feats_fixed = all_data[skewed_feats_over].apply(lambda x: x.skew()).sort_values(ascending = False)



#グラフ化

plt.figure(figsize=(20,10))

plt.xticks(rotation='90')

sns.barplot(x=skewed_feats_fixed.index, y=skewed_feats_fixed)
#特徴量に1部屋あたりの面積を追加

all_data["FeetPerRoom"] =  all_data["TotalSF"]/all_data["TotRmsAbvGrd"]



#その他有効そうなものを追加する



#建築した年とリフォームした年の合計

all_data['YearBuiltAndRemod']=all_data['YearBuilt']+all_data['YearRemodAdd']



#バスルームの合計面積

all_data['Total_Bathrooms'] = (all_data['FullBath'] + (0.5 * all_data['HalfBath']) +

                               all_data['BsmtFullBath'] + (0.5 * all_data['BsmtHalfBath']))



#縁側の合計面積

all_data['Total_porch_sf'] = (all_data['OpenPorchSF'] + all_data['3SsnPorch'] +

                              all_data['EnclosedPorch'] + all_data['ScreenPorch'] +

                              all_data['WoodDeckSF'])



#プールの有無

all_data['haspool'] = all_data['PoolArea'].apply(lambda x: 1 if x > 0 else 0)



#2階の有無

all_data['has2ndfloor'] = all_data['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)



#ガレージの有無

all_data['hasgarage'] = all_data['GarageArea'].apply(lambda x: 1 if x > 0 else 0)



#地下室の有無

all_data['hasbsmt'] = all_data['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)



#暖炉の有無

all_data['hasfireplace'] = all_data['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
#各カラムのデータ型を確認

all_data.dtypes.value_counts()
#カテゴリ変数となっているカラムを取り出す

cal_list = all_data.dtypes[all_data.dtypes=="object"].index.tolist()



#学習データにおけるカテゴリ変数のデータ数を確認

train_x[cal_list].info()
#カテゴリ変数をget_dummiesによるone-hot-encodingを行う

all_data = pd.get_dummies(all_data,columns=cal_list)



#サイズを確認

all_data.shape
#学習データとテストデータに再分割

train_x = all_data.iloc[:train_x.shape[0],:].reset_index(drop=True)

test_x = all_data.iloc[train_x.shape[0]:,:].reset_index(drop=True)



#サイズを確認

print("train_x: "+str(train_x.shape))

print("test_x: "+str(test_x.shape))
scaler = StandardScaler()  #スケーリング

param_grid = [0.001, 0.01, 0.1, 1.0, 10.0,100.0,1000.0] #パラメータグリッド

cnt = 0

for alpha in param_grid:

    ls = ElasticNet(alpha=alpha, l1_ratio = 0.7) #ElasticNet回帰モデル

    pipeline = make_pipeline(scaler, ls) #パイプライン生成

    X_train, X_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.3, random_state=0)

    pipeline.fit(X_train,y_train)

    train_rmse = np.sqrt(mean_squared_error(y_train, pipeline.predict(X_train)))

    test_rmse = np.sqrt(mean_squared_error(y_test, pipeline.predict(X_test)))

    if cnt == 0:

        best_score = test_rmse

        best_estimator = pipeline

        best_param = alpha

    elif best_score > test_rmse:

        best_score = test_rmse

        best_estimator = pipeline

        best_param = alpha

    else:

        pass

    cnt = cnt + 1

    

print('alpha : ' + str(best_param))

print('test score is : ' +str(best_score))
ls = ElasticNet(alpha = 0.01)

pipeline = make_pipeline(scaler, ls)

pipeline.fit(train_x,train_y)

test_SalePrice = pd.DataFrame(np.exp(pipeline.predict(test_x)),columns=['SalePrice'])

test_Id = pd.DataFrame(test_ID,columns=['Id'])

pd.concat([test_Id, test_SalePrice],axis=1).to_csv('submission.csv',index=False)