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
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
train = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")
test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")
print(train.shape)
print(test.shape)
# 最大カラム数を100に拡張(デフォルトだと省略されてしまうので)
pd.set_option('display.max_columns', 100)
train.head()
train.info()
train.describe()
train.describe(include="O")
train["SalePrice"].describe()

#目的変数である家の価格のヒストグラムを表示する
sns.distplot(train['SalePrice']);
from scipy import stats
res = stats.probplot(train['SalePrice'], plot=plt)
#歪度と尖度を計算
#歪度:分布が正規分布からどれだけ歪んでいるかを表す統計量で、左右対称性を示す指標
#尖度:分布が正規分布からどれだけ尖っているかを表す統計量で、山の尖り度と裾の広がり度

print("歪度: %f" % train['SalePrice'].skew())
print("尖度: %f" % train['SalePrice'].kurt())
#1階のフロアの面積(1stFlrSF)
g=sns.FacetGrid(train)
g=g.map(sns.distplot,"1stFlrSF")
g.add_legend()

#2階のフロアの面積(2ndFlrSF)
g=sns.FacetGrid(train)
g=g.map(sns.distplot,"2ndFlrSF")

#地下室の面積(TotalBsmtSF)
g=sns.FacetGrid(train)
g=g.map(sns.distplot,"TotalBsmtSF")

#物件の広さを合計した変数を作成
train["TotalSF"] = train["1stFlrSF"] + train["2ndFlrSF"] + train["TotalBsmtSF"]
test["TotalSF"] = test["1stFlrSF"] + test["2ndFlrSF"] + test["TotalBsmtSF"]

#元の変数を削除
train = train.drop(["1stFlrSF","2ndFlrSF","TotalBsmtSF"],axis=1)


#物件の広さと物件価格の散布図を作成
plt.figure(figsize=(20, 10))
plt.scatter(train["TotalSF"],train["SalePrice"])
plt.xlabel("TotalSF")
plt.ylabel("SalePrice")

#外れ値を除外する
train = train.drop(train[(train['TotalSF']>7500) & (train['SalePrice']<300000)].index)

#物件の広さと物件価格の散布図を作成
plt.figure(figsize=(20, 10))
plt.scatter(train["TotalSF"],train["SalePrice"])
plt.xlabel("TotalSF")
plt.ylabel("SalePrice")

#築年数と物件価格の散布図を作成
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

k = 10
df = train
corrmat = df.corr()
cols = corrmat.nlargest(k, "SalePrice")["SalePrice"].index
cm = np.corrcoef(df[cols].values.T)
fig, ax = plt.subplots(figsize=(12, 10))
sns.set(font_scale=1.2)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt=".2f", annot_kws={"size": 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
fig.savefig("figure4.png")


#選択された項目のみを使用する
train = train[cols]

#多重共線性を起こす可能性のある変数を削除
train = train.drop(["TotRmsAbvGrd","GarageArea"],axis = 1)
test = test.drop(["TotRmsAbvGrd","GarageArea"],axis = 1)
train.head()

sns.pairplot(train,y_vars=['SalePrice'],x_vars=['TotalSF','OverallQual','GrLivArea','GarageCars','FullBath','YearBuilt','YearRemodAdd'])

#学習データを目的変数とそれ以外に分ける
train_X = train.drop("SalePrice",axis=1)
train_y = train["SalePrice"]

#テストデータを学習データのカラムのみにする 
tmp_cols = train_X.columns
test_X = test[tmp_cols]

#それぞれのデータのサイズを確認
print("train_X: "+str(train_X.shape))
print("train_y: "+str(train_y.shape))
print("test_X: "+str(test_X.shape))
#データの欠損値を確認する
#学習データ 
print(train_X.isnull().sum())
print('--------------------') #区切り文字
#テストデータ
print(test_X.isnull().sum())

#学習データの欠損値を平均値で置き換える
test_X = test_X.fillna(test_X.mean())
test_X.isnull().sum()
#目的変数の対数log(x+1)をとる
train_y = np.log1p(train_y)

#分布を可視化
plt.figure(figsize=(10, 5))
sns.distplot(train_y)
fig = plt.figure()
res = stats.probplot(train_y, plot=plt)

#訓練データとモデル評価用データに分けるライブラリ
from sklearn.model_selection import train_test_split

#ホールドアウト法により、学習データとテストデータに分割 
(X_train, X_test, y_train, y_test) = train_test_split(train_X, train_y , test_size = 0.3 , random_state = 0)

print("X_train: "+str(X_train.shape))
print("X_test: "+str(X_test.shape))
print("y_train: "+str(y_train.shape))
print("y_test: "+str(y_test.shape))

from sklearn import linear_model

# 線形回帰モデルを構築
clf = linear_model.LinearRegression()
 
# 単回帰用に説明変数を一つとする    
X_train_1 = X_train[["YearBuilt"]] 
X_test_1 = X_test[["YearBuilt"]]  
    
# モデル学習
clf.fit(X_train_1, y_train)
 
# 学習結果の回帰係数
print('回帰係数： {}'.format(clf.coef_))
 
# 学習結果の切片 (誤差)
print('切片： {}'.format(clf.intercept_))

# テストデータにて予測
y_pred_log = clf.predict(X_test_1)

#目的変数を対数変換しているので元に戻す
y_pred =np.exp(y_pred_log) 
print(y_pred)

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

#RMSE(平均平方二乗誤差)
print(np.sqrt(mean_squared_error(np.exp(y_test), y_pred)))

#MAE(平均絶対誤差)
print(mean_absolute_error(np.exp(y_test), y_pred))

# 決定係数
print(r2_score(np.exp(y_test), y_pred))

# 線形回帰モデルを使用
clf = linear_model.LinearRegression()
 
# モデル学習
clf.fit(X_train, y_train)
  
# 偏回帰係数
print('回帰係数： {}'.format(pd.DataFrame({"Name":X_train.columns,
                    "Coefficients":clf.coef_}).sort_values(by='Coefficients') ))

# 学習結果の切片 (誤差)
print('切片： {}'.format(clf.intercept_))

# テストデータにて予測
y_pred_log = clf.predict(X_test)

#目的変数を対数変換しているので元に戻す
y_pred =np.exp(y_pred_log) 

#RMSE(平均平方二乗誤差)
print(np.sqrt(mean_squared_error(np.exp(y_test), y_pred)))

#MAE(平均絶対誤差)
print(mean_absolute_error(np.exp(y_test), y_pred))

# 決定係数
print(r2_score(np.exp(y_test), y_pred))

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

#モデルの構築
rfr = RandomForestRegressor(random_state=0)

#学習データにて学習
rfr.fit(X_train, y_train)

# テストデータにて予測
y_pred_log = rfr.predict(X_test)

#目的変数を対数変換しているので元に戻す
y_pred =np.exp(y_pred_log)

#RMSE(平均平方二乗誤差)
print(np.sqrt(mean_squared_error(np.exp(y_test), y_pred)))

#MAE(平均絶対誤差)
print(mean_absolute_error(np.exp(y_test), y_pred))

# 決定係数
print(r2_score(np.exp(y_test), y_pred))

plt.figure(figsize=(20,10))
plt.barh(
    X_train.columns[np.argsort(rfr.feature_importances_)],
    rfr.feature_importances_[np.argsort(rfr.feature_importances_)],
     label='RandomForestRegressor'
 )
plt.title('RandomForestRegressor feature importance')

# 【参考】スコア提出
# hon_pred_log = rfr.predict(test_X)
# my_submission = pd.DataFrame()
# my_submission["Id"] = test['Id']
# my_submission["SalePrice"] = np.exp(hon_pred_log)
# my_submission.to_csv('submission.csv', index=False)
