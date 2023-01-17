import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PowerTransformer
# データの読み込み
train1 = pd.read_csv("/Users/hirumakazuya/Downloads/train.csv")
test1 = pd.read_csv("/Users/hirumakazuya/Downloads/test.csv")
print(train1)
print(test1)
train1.columns
##目的変数である家の価格の要約統計量を表示する
train1["SalePrice"].describe()
#目的変数である家の価格のヒストグラムを表示する
plt.figure(figsize=(20, 10))
sns.distplot(train1['SalePrice'])
#歪度と尖度を計算
print("歪度: %f" % train1['SalePrice'].skew())
print("尖度: %f" % train1['SalePrice'].kurt())
#物件の広さを合計した変数を作成
train1["TotalSF"] = train1["1stFlrSF"] + train1["2ndFlrSF"] + train1["TotalBsmtSF"]
test1["TotalSF"] = test1["1stFlrSF"] + test1["2ndFlrSF"] + test1["TotalBsmtSF"]

#合計した変数を削除
# train.drop(['1stFlrSF', '2ndFlrSF', 'TotalBsmtSF'], axis = 1, inplace = True)
# test_x.drop(['1stFlrSF', '2ndFlrSF', 'TotalBsmtSF'], axis = 1, inplace = True)

#物件の広さと物件価格の散布図を作成
plt.figure(figsize=(20, 10))
plt.scatter(train1["TotalSF"],train1["SalePrice"])
plt.xlabel("TotalSF")
plt.ylabel("SalePrice")
#外れ値を除外する
train1 = train1.drop(train1[(train1['TotalSF']>7500) & (train1['SalePrice']<300000)].index)

#物件の広さと物件価格の散布図を作成
plt.figure(figsize=(20, 10))
plt.scatter(train1["TotalSF"],train1["SalePrice"])
plt.xlabel("TotalSF")
plt.ylabel("SalePrice")
#築年数と物件価格の散布図を作成
#plt.scatter(train["YearBuilt"],train["SalePrice"],color = "#e41a1c")
#plt.xlabel("YearBuilt")
#plt.ylabel("SalePrice")

data = pd.concat([train1["YearBuilt"],train1["SalePrice"]],axis=1)

plt.figure(figsize=(20, 10))
plt.xticks(rotation='90')
sns.boxplot(x="YearBuilt",y="SalePrice",data=data)
#外れ値を除外する
train1 = train1.drop(train1[(train1['YearBuilt']<2000) & (train1['SalePrice']>600000)].index)

#グラフを描画する
data = pd.concat([train1["YearBuilt"],train1["SalePrice"]],axis=1)

plt.figure(figsize=(20, 10))
plt.xticks(rotation='90')
sns.boxplot(x="YearBuilt",y="SalePrice",data=data)
#家の材質・完成度と物件価格の散布図を作成
plt.figure(figsize=(20, 10))
plt.scatter(train1["OverallQual"],train1["SalePrice"])
plt.xlabel("OverallQual")
plt.ylabel("SalePrice")
#外れ値を除外する
train1 = train1.drop(train1[(train1['OverallQual']<5) & (train1['SalePrice']>200000)].index)
train1 = train1.drop(train1[(train1['OverallQual']<10) & (train1['SalePrice']>500000)].index)
#グラフを描画する
plt.figure(figsize=(20, 10))
plt.scatter(train1["OverallQual"],train1["SalePrice"])
plt.xlabel("OverallQual")
plt.ylabel("SalePrice")
#学習データを目的変数とそれ以外に分ける
train_x = train1.drop("SalePrice",axis=1)
train_y = train1["SalePrice"]

all_data = pd.concat([train_x,test1],axis=0,sort=True)

#IDのカラムは不必要なので別の変数に格納
train_ID = train1['Id']
test_ID = test1['Id']

all_data.drop("Id", axis = 1, inplace = True)

#それぞれのデータのサイズを確認
print("train_x: "+str(train_x.shape))
print("train_y: "+str(train_y.shape))
print("test1: "+str(test1.shape))
print("all_data: "+str(all_data.shape))
#データの欠損値を確認する
all_data_na = all_data.isnull().sum()[all_data.isnull().sum()>0].sort_values(ascending=False)
all_data_na
#欠損値の数をグラフ化
plt.figure(figsize=(20,10))
plt.xticks(rotation='90')
sns.barplot(x=all_data_na.index, y=all_data_na)
# 欠損値があるカラムをリスト化
na_col_list = all_data.isnull().sum()[all_data.isnull().sum()>0].index.tolist()

#欠損があるカラムのデータ型を確認
all_data[na_col_list].dtypes.sort_values()
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
#各カラムのデータ型を確認
all_data.dtypes.value_counts()
#カテゴリ変数となっているカラムを取り出す
cal_list = all_data.dtypes[all_data.dtypes=="object"].index.tolist()
train_x[cal_list].info()
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import log_loss
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
# データの分割
train_x, valid_x, train_y, valid_y = train_test_split(
        train_x,
        train_y,
        test_size=0.3,
        random_state=0)
#特徴量と目的変数をxgboostのデータ構造に変換する
dtrain = xgb.DMatrix(train_x, label=train_y)
dvalid = xgb.DMatrix(valid_x,label=valid_y)
#パラメータを指定してGBDT
num_round = 5000
evallist = [(dvalid, 'eval'), (dtrain, 'train')]

evals_result = {}

#パラメータ
param = {
            'max_depth': 3,
            'eta': 0.01,
            'objective': 'reg:squarederror',
}

#学習の実行
bst = xgb.train(
                        param, dtrain,
                        num_round,
                        evallist,
                        evals_result=evals_result,
                        # 一定ラウンド回しても改善が見込めない場合は学習を打ち切る
                        early_stopping_rounds=1000
                        )

#bst = xgb.train(param, dtrain, num_round)

#[4999]	eval-rmse:0.124586	train-rmse:0.075149
#[4999]	eval-rmse:0.118845	train-rmse:0.073414
