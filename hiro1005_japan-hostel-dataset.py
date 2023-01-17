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
%matplotlib inline

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec

import seaborn as sns 



from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split

from sklearn.linear_model import RANSACRegressor

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Ridge

from sklearn.linear_model import LassoCV

from sklearn.linear_model import ElasticNet

from sklearn.kernel_ridge import KernelRidge

from sklearn.svm import SVR

from sklearn.svm import LinearSVR

from sklearn.tree import DecisionTreeRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import GradientBoostingRegressor

from xgboost.sklearn import XGBRegressor

from lightgbm import LGBMRegressor

from catboost import CatBoostRegressor

from sklearn.linear_model import SGDRegressor

from sklearn.ensemble import VotingRegressor

hostel = pd.read_csv('/kaggle/input/hostel-world-dataset/Hostel.csv', header=0)

hostel

hostel['City'].value_counts()
hostel['rating.band'].value_counts()
each_city = pd.get_dummies(hostel['City'])

rating = pd.get_dummies(hostel['rating.band'])



del hostel['rating.band']

del hostel['City']







def Missing_table(df):

    # null_val = df.isnull().sum()

    null_val = df.isnull().sum()[df.isnull().sum()>0].sort_values(ascending=False)

    percent = 100 * null_val/len(df)

    na_col_list = df.isnull().sum()[df.isnull().sum()>0].index.tolist() # 欠損を含むカラムをリスト化

    list_type = df[na_col_list].dtypes.sort_values(ascending=False) #データ型

    Missing_table = pd.concat([null_val, percent, list_type], axis = 1)

    missing_table_len = Missing_table.rename(

    columns = {0:'欠損値', 1:'欠損値の割合(%)', 2:'type'})

    return missing_table_len.sort_values(by=['欠損値'], ascending=False)



Missing_table(hostel)

# いくつか欠損値がある

pd.set_option('display.max_rows', 2000)

hostel

# 欠損値を含むすべての行を削除（inplace=Trueをつけなければ、元データに反映されない）

hostel.dropna(inplace=True)





hostel.sort_values('price.from', ascending=False)

#あきらかに、289, 316だけ価格が外れているので、これは削除したほうがいい

hostel.drop(index=[289, 316], inplace=True)



# 名前・緯度などの、価格に関係なさそうな列は削除する

del hostel['Unnamed: 0']

del hostel['hostel.name']

del hostel['lon']

del hostel['lat']





# 全データを結合

total_hostel = pd.concat([hostel, each_city, rating], axis=1)



# 全データ結合後は、またNaNなどが出てくるので、再度削除する

total_hostel.dropna(inplace=True)





# 全データ型を確認

total_hostel.dtypes.sort_values()

# Distanceがobjectになっているので、数値に変換する

# km from city centre の余計な文字列を削除

total_hostel['Distance'] =  total_hostel['Distance'].str.replace("km from city centre", "")

# objectをfloatに変換する

total_hostel['Distance'] = total_hostel['Distance'].astype('float')

# EDA

total_hostel.hist(figsize = (12,12))

fig, ax = plt.subplots(figsize=(15,15))

sns.heatmap(total_hostel.corr(),annot=True, center=0, square=True, linewidths=0.1, vmax=1.0, linecolor='white', cmap="RdBu")

plt.title('Japan Hostel Dataset Correlation of Features', fontsize = 20)

plt.xlabel('x-axis', fontsize = 15)

plt.ylabel('y-axis', fontsize = 15)



plt.figure(figsize=(30, 40))

sns.countplot(x='price.from', data = total_hostel)

plt.figure(figsize=(15, 15))

plt.rcParams["font.size"] = 20

x = total_hostel['price.from']

y = total_hostel['Distance']

 

# 散布図を描画

plt.scatter(x, y)

plt.title("Relatonship with Distance and Price")

plt.xlabel("Price")

plt.ylabel("Distance")

plt.grid(True)

plt.figure(figsize=(15, 15))

sns.distplot(total_hostel[total_hostel['Fukuoka-City']==1]['price.from'],kde=False,rug=False,bins=10,label='Fukuoka-City')

sns.distplot(total_hostel[total_hostel['Hiroshima']==1]['price.from'],kde=False,rug=False,bins=10,label='Hiroshima')

sns.distplot(total_hostel[total_hostel['Kyoto']==1]['price.from'],kde=False,rug=False,bins=10,label='Kyoto')

sns.distplot(total_hostel[total_hostel['Osaka']==1]['price.from'],kde=False,rug=False,bins=10,label='Osaka')

sns.distplot(total_hostel[total_hostel['Tokyo']==1]['price.from'],kde=False,rug=False,bins=10,label='Tokyo')

plt.legend()

train_feature = total_hostel.drop(columns='price.from')

train_target = total_hostel['price.from']



X_train, X_test, y_train, y_test = train_test_split(train_feature, train_target, test_size=0.2, random_state=0, shuffle=True)



# 有効な特微量を探す（SelectKBestの場合）

from sklearn.feature_selection import SelectKBest, f_regression

# 特に重要な4つの特徴量のみを探すように設定してみる

selector = SelectKBest(score_func=f_regression, k=4) 

selector.fit(train_feature, train_target)

mask_SelectKBest = selector.get_support()    # 各特徴量を選択したか否かのmaskを取得



# 有効な特微量を探す（SelectPercentileの場合）

from sklearn.feature_selection import SelectPercentile, f_regression

# 特徴量のうち40%を選択

selector = SelectPercentile(score_func=f_regression, percentile=40) 

selector.fit(train_feature, train_target)

mask_SelectPercentile = selector.get_support()



# 有効な特微量を探す（モデルベース選択の場合：SelectFromModel）

from sklearn.feature_selection import SelectFromModel

from sklearn.ensemble import RandomForestRegressor

# estimator として RandomForestRegressor を使用。重要度が median 以上のものを選択

selector = SelectFromModel(RandomForestRegressor(n_estimators=100, random_state=42), threshold="median")    

selector.fit(train_feature, train_target)

mask_SelectFromModel = selector.get_support()



# 有効な特微量を探す（RFE：再帰的特徴量削減 : n_features_to_select）

from sklearn.feature_selection import RFE

from sklearn.ensemble import RandomForestRegressor

# estimator として RandomForestRegressor を使用。特徴量を2個選択させる

selector = RFE(RandomForestRegressor(n_estimators=100, random_state=42), n_features_to_select=2)

selector.fit(train_feature, train_target)

mask_RFE = selector.get_support()



print(train_feature.columns)

print(mask_SelectKBest)

print(mask_SelectPercentile)

print(mask_SelectFromModel)

print(mask_RFE)

import warnings

warnings.filterwarnings('ignore')



# RandomForest==============



rf = RandomForestRegressor(n_estimators=200, max_depth=5, max_features=0.5,  verbose=True, random_state=0, n_jobs=-1) # RandomForest のオブジェクトを用意する

rf.fit(X_train, y_train)

print('='*20)

print('RandomForestRegressor')

print(f'accuracy of train set: {rf.score(X_train, y_train)}')

print(f'accuracy of test set: {rf.score(X_test, y_test)}')



# SVR（Support Vector Regression）==============

# ※[LibSVM]や[LibLinear]は台湾国立大学の方で開発されたらしくどうしてもその表示が入るようになっている



svr = SVR(verbose=True)

svr.fit(X_train, y_train)

print('='*20)

print('SVR')

print(f'accuracy of train set: {svr.score(X_train, y_train)}')

print(f'accuracy of test set: {svr.score(X_test, y_test)}')



# LinearSVR==============



lsvr = LinearSVR(verbose=True, random_state=0)

lsvr.fit(X_train, y_train)

print('='*20)

print('LinearSVR')

print(f'accuracy of train set: {lsvr.score(X_train, y_train)}')

print(f'accuracy of test set: {lsvr.score(X_test, y_test)}')



# SGDRegressor==============



sgd = SGDRegressor(verbose=0, random_state=0)

sgd.fit(X_train, y_train)

print('='*20)

print('SGDRegressor')

print(f'accuracy of train set: {sgd.score(X_train, y_train)}')

print(f'accuracy of test set: {sgd.score(X_test, y_test)}')



# k-近傍法（k-NN）==============



knn = KNeighborsRegressor()

knn.fit(X_train, y_train)

print('='*20)

print('KNeighborsRegressor')

print(f'accuracy of train set: {knn.score(X_train, y_train)}')

print(f'accuracy of test set: {knn.score(X_test, y_test)}')



# 決定木==============



decisiontree = DecisionTreeRegressor(max_depth=3, random_state=0)

decisiontree.fit(X_train, y_train)

print('='*20)

print('DecisionTreeRegressor')

print(f'accuracy of train set: {decisiontree.score(X_train, y_train)}')

print(f'accuracy of test set: {decisiontree.score(X_test, y_test)}')



# LinearRegression (線形回帰)==============



lr = LinearRegression()

lr.fit(X_train, y_train)

print('='*20)

print('LinearRegression')

print(f'accuracy of train set: {lr.score(X_train, y_train)}')

print(f'accuracy of test set: {lr.score(X_test, y_test)}')

# 回帰係数とは、回帰分析において座標平面上で回帰式で表される直線の傾き。 原因となる変数x（説明変数）と結果となる変数y（目的変数）の平均的な関係を、一次式y＝ax＋bで表したときの、係数aを指す。

print("回帰係数:",lr.coef_)

print("切片:",lr.intercept_)
