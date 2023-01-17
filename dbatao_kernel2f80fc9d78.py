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
import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

import seaborn as sns

from sklearn.preprocessing import PowerTransformer
#データの読み込み



train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv') #訓練データ

test_x = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv') #テストデータ



#学習データの変数を確認

train.columns
#目的変数である家の価格の要約統計量を表示する

train["SalePrice"].describe()
#50パーセントタイルが16万なのに対して平均が18万となっている、かつMAXが75万と大きいので正規分布にはなってない気がします。ヒストグラムを見てみます。
#目的変数である家の価格のヒストグラムを表示する

plt.figure(figsize=(20, 10))

sns.distplot(train['SalePrice'])
#やはり正規分布になっていませんね。目的変数の分布は正規分布になっていないとモデルの予測精度に影響が出てしまうので、これはあとで対数変換する必要がありそうです。
#歪度と尖度を計算

print("歪度: %f" % train['SalePrice'].skew())

print("尖度: %f" % train['SalePrice'].kurt())
#歪度が正の値なので、右裾が長い（≒左に偏っている）分布であること、さらに尖度が正の値のため正規分布よりもだいぶ尖った（平均付近にデータが集中している）分布であることがわかります。
#ここで与えられた特徴量をみると



#物件の大きさ（広さ）

#築年数

#家の材質と完成度

#あたりがぱっとみで物件の価格に影響しそうなので、目的変数との関係を見てみます。
#物件の広さを合計した変数を作成

train["TotalSF"] = train["1stFlrSF"] + train["2ndFlrSF"] + train["TotalBsmtSF"]

test_x["TotalSF"] = test_x["1stFlrSF"] + test_x["2ndFlrSF"] + test_x["TotalBsmtSF"]



#物件の広さと物件価格の散布図を作成

plt.figure(figsize=(20, 10))

plt.scatter(train["TotalSF"],train["SalePrice"])

plt.xlabel("TotalSF")

plt.ylabel("SalePrice")
#若干外れ値がありますが、相関しているように見えます。やはり物件が大きくなるほど物件価格も高くなることがわかります。
#外れ値を除外する

train = train.drop(train[(train['TotalSF']>7500) & (train['SalePrice']<300000)].index)



#物件の広さと物件価格の散布図を作成

plt.figure(figsize=(20, 10))

plt.scatter(train["TotalSF"],train["SalePrice"])

plt.xlabel("TotalSF")

plt.ylabel("SalePrice")
#築年数と物件価格の散布図を作成

#plt.scatter(train["YearBuilt"],train["SalePrice"],color = "#e41a1c")

#plt.xlabel("YearBuilt")

#plt.ylabel("SalePrice")



data = pd.concat([train["YearBuilt"],train["SalePrice"]],axis=1)



plt.figure(figsize=(20, 10))

plt.xticks(rotation='90')

sns.boxplot(x="YearBuilt",y="SalePrice",data=data)
#微妙ですが築年数が新しいほど物件価格が高くなる傾向はありそうです。

#こちらも外れ値があるので除外します。
#外れ値を除外する

train = train.drop(train[(train['YearBuilt']<2000) & (train['SalePrice']>600000)].index)



#グラフを描画する

data = pd.concat([train["YearBuilt"],train["SalePrice"]],axis=1)



plt.figure(figsize=(20, 10))

plt.xticks(rotation='90')

sns.boxplot(x="YearBuilt",y="SalePrice",data=data)
#さらに以下のような特徴量があります。これを家の材質と完成度と訳すのがよいのかわかりませんが価格には影響与えそうなので可視化してみます。
#家の材質・完成度と物件価格の散布図を作成

plt.figure(figsize=(20, 10))

plt.scatter(train["OverallQual"],train["SalePrice"])

plt.xlabel("OverallQual")

plt.ylabel("SalePrice")
#クオリティが上がるほど価格が上がっているように見えます。
#外れ値を除外する

train = train.drop(train[(train['OverallQual']<5) & (train['SalePrice']>200000)].index)

train = train.drop(train[(train['OverallQual']<10) & (train['SalePrice']>500000)].index)



#グラフを描画する

plt.figure(figsize=(20, 10))

plt.scatter(train["OverallQual"],train["SalePrice"])

plt.xlabel("OverallQual")

plt.ylabel("SalePrice")
#前処理をいっぺんに行うために、学習データとテストデータをマージします。 さらに学習データに関しては目的変数である「SalesPrice」が含まれているのでこちらは切り出しておきます。
#学習データを目的変数とそれ以外に分ける

train_x = train.drop("SalePrice",axis=1)

train_y = train["SalePrice"]



#学習データとテストデータを統合

all_data = pd.concat([train_x,test_x],axis=0,sort=True)



#IDのカラムは不必要なので別の変数に格納

train_ID = train['Id']

test_ID = test_x['Id']



all_data.drop("Id", axis = 1, inplace = True)



#それぞれのデータのサイズを確認

print("train_x: "+str(train_x.shape))

print("train_y: "+str(train_y.shape))

print("test_x: "+str(test_x.shape))

print("all_data: "+str(all_data.shape))
#続いて欠損値を処理していきたいと思います。まずはどのくらい欠損値があるのかを確認します。
#データの欠損値を確認する

all_data_na = all_data.isnull().sum()[all_data.isnull().sum()>0].sort_values(ascending=False)

all_data_na
#めちゃくちゃありますね、、、グラフ化してみます。
#欠損値の数をグラフ化

plt.figure(figsize=(20,10))

plt.xticks(rotation='90')

sns.barplot(x=all_data_na.index, y=all_data_na)
#欠損値については一括で削除するか、平均値で置換してしまいたくなってしまいますが、

#きちんと欠損値が多い変数が何を表すものか見てみます。



#PoolQC: Pool quality

#備え付けられているプールの質を表す。プールがない場合にはNAとなる。

#MiscFeature: Miscellaneous feature not covered in other categories

#その他の備え付けられている設備を表す。エレベータやテニスコートなど。特にない場合はNAとなる。

#Alley: Type of alley access to property

#物件にアクセスするための道の種類（砂利なのか舗装されているのか）を表す。該当しない場合はNAとなる

#Fence: Fence quality

#フェンスの質を表す。フェンスがない場合はNAとなる。

#FireplaceQu: Fireplace quality

#暖炉の品質を表す。暖炉がない場合はNAとなる。

#LotFrontage: Linear feet of street connected to property

#物件に隣接した道路の長さ。

#欠損値が多い変数に関しては、データが欠損しているというわけではなく、欠損＝そもそもその設備がないことを表しているようですね。

#次に欠損値がある変数のデータ型を確認します。
# 欠損値があるカラムをリスト化

na_col_list = all_data.isnull().sum()[all_data.isnull().sum()>0].index.tolist()



#欠損があるカラムのデータ型を確認

all_data[na_col_list].dtypes.sort_values()
#floatとobjectですね。float型の場合は0、objectの場合は”None”で置換することにしましょう。

#ただし、隣接した道路の長さ（LotFrontage）に関しては同じ地区と他の物件と同じと思われるので、同じ地区の中央値をとることにします。
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
#さきほどはカテゴリ変数を処理しましたが、続いては数値変数です。

#数値変数は基本はそのままモデルの入力に使えますが、データ上は数値であっても値の大きさや順番に意味のないものはカテゴリ変数として扱うべきです。

#もう一度データの定義を見てみると、以下の変数は数値変数ではなくカテゴリ変数として扱ったほうが良さそうです。



#MSSubClass: Identifies the type of dwelling involved in the sale

#住宅の種類を表す。数値はどの種類に当てはまるかを表すだけで大きさや順序に意味はない。

#YrSold: Year Sold (YYYY)

#販売年

#MoSold: Month Sold (MM)

#販売月
# カテゴリ変数に変換する

all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)

all_data['YrSold'] = all_data['YrSold'].astype(str)

all_data['MoSold'] = all_data['MoSold'].astype(str)
#探索的データ分析のフェーズで目的変数であるSalesPriceが正規分布になっていないことがわかりました。

#こちらを正規分布に変換します。
#目的変数の対数log(x+1)をとる

train_y = np.log1p(train_y)



#分布を可視化

plt.figure(figsize=(20, 10))

sns.distplot(train_y)
#同様に説明変数に関しても正規分布にしたがっていないものは対数変換していきます。

#各説明変数に対して歪度を計算し、0.5よりも大きい場合は対数変換することにします。
#数値の説明変数のリストを作成

num_feats = all_data.dtypes[all_data.dtypes != "object" ].index



#各説明変数の歪度を計算

skewed_feats = all_data[num_feats].apply(lambda x: x.skew()).sort_values(ascending = False)



#グラフ化

plt.figure(figsize=(20,10))

plt.xticks(rotation='90')

sns.barplot(x=skewed_feats.index, y=skewed_feats)
#ちなみにBox-Cox変換は0以下の値をとる変数には使用できないため、各変数の最小値を見てみます。
#歪度の絶対値が0.5より大きい変数だけに絞る

skewed_feats_over = skewed_feats[abs(skewed_feats) > 0.5].index



#各変数の最小値を表示

for i in skewed_feats_over:

    print(min(all_data[i]))
#負の値はないですが、0が含まれる変数がありますね、、、

#今回はBox-Cox変換ではなく、０以下の値を持つ変数にも適用可能なYeo-Johnson変換を使いたいと思います。
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
#同じくSalesPriceを予測する「Zillow Prize: Zillow’s Home Value Prediction」で、物件の面積を部屋数で割った１部屋当たりの面積という特徴量を追加したことで精度が上がった、ということがあったそうなので今回も追加してみます。
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
#カテゴリ変数は46つあります。



#学習データには存在せず、テストデータのみに存在するカテゴリ変数が存在するとモデルがそのカテゴリを学習できず、予測値がおかしくなる可能性があります。

#テストデータのみに存在するカテゴリ変数が存在しないかを確認しましょう。
#カテゴリ変数となっているカラムを取り出す

cal_list = all_data.dtypes[all_data.dtypes=="object"].index.tolist()



#学習データにおけるカテゴリ変数のデータ数を確認

train_x[cal_list].info()
#学習データの中に数が0となっているカテゴリ変数はないようですのでそのまま進めていきます。
#カテゴリ変数のエンコーディング方法はさまざまありますのが、今回はone-hot-encodingによってハンドリングします。

#one-hot-encodingでは、各カテゴリ変数を（0,1）の二値変数をそれぞれ作成します。これらの二値変数は「ダミー変数」と呼ばれます。



#pandasのget_dummies関数でone-hot-encodingを行います。
#カテゴリ変数をget_dummiesによるone-hot-encodingを行う

all_data = pd.get_dummies(all_data,columns=cal_list)



#サイズを確認

all_data.shape
#前処理とエンコーディングが終わったのでデータを学習データとテストデータに分割しておきます。
#学習データとテストデータに再分割

train_x = all_data.iloc[:train_x.shape[0],:].reset_index(drop=True)

test_x = all_data.iloc[train_x.shape[0]:,:].reset_index(drop=True)



#サイズを確認

print("train_x: "+str(train_x.shape))

print("test_x: "+str(test_x.shape))
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor

from sklearn.model_selection import KFold, cross_val_score, train_test_split

from sklearn.metrics import mean_squared_error

import xgboost as xgb

from sklearn.model_selection import GridSearchCV

import lightgbm as lgb
#モデルの評価用に学習データを学習に使用するデータと評価用データ（バリデーションデータ）に分割します。
# データの分割

train_x, valid_x, train_y, valid_y = train_test_split(

        train_x,

        train_y,

        test_size=0.3,

        random_state=0)
#最近、kaggleで人気だと言われているGBDT（勾配ブースティング木）でモデルを作成します。GBDTはランダムフォレストと勾配ブースティング（Gradient Boosting）を組み合わせたアンサンブルです。GBDTには以下の特徴があります。



#特徴量は数値である必要がある

#欠損値の補完が不要
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
#学習曲線を可視化する

plt.figure(figsize=(20, 10))

train_metric = evals_result['train']['rmse']

plt.plot(train_metric, label='train rmse')

eval_metric = evals_result['eval']['rmse']

plt.plot(eval_metric, label='eval rmse')

plt.grid()

plt.legend()

plt.xlabel('rounds')

plt.ylabel('rmse')

plt.ylim(0, 0.3)

plt.show()
#特徴量ごとの重要度を可視化する

ax = xgb.plot_importance(bst)

fig = ax.figure

fig.set_size_inches(10, 30)
dtest = xgb.DMatrix(test_x)

my_submission = pd.DataFrame()

my_submission["Id"] = test_ID

my_submission["SalePrice"] = np.exp(bst.predict(dtest))

# you could use any filename. We choose submission here

my_submission.to_csv('submission.csv', index=False)