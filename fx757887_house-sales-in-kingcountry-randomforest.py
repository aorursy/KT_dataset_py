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
#データ処理の基本的なライブラリインポート

import pandas as pd

import numpy as np



#データ可視化ライブラリ

import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec

import seaborn as sns



#機械学習ライブラリ

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error

from sklearn.tree import DecisionTreeRegressor



from sklearn.model_selection import GridSearchCV, train_test_split



#XGBoost

import xgboost as xgb

from xgboost import XGBClassifier







#決定木の可視化

from sklearn import tree

import graphviz



#その他設定

pd.set_option("max_columns",35)

pd.set_option("max_rows",600)

sns.set_style("darkgrid")
#ファイル読込

kchouse=pd.read_csv("/kaggle/input/housesalesprediction/kc_house_data.csv")
#先頭確認

kchouse.head()
#基本統計量確認

kchouse.describe()
kchouse.columns
kchouse.info()
#不要なカラム削除

kchouse=kchouse.drop(["id","date"],axis=1)
# 寝室の数（bedrooms）を基軸にグルーピング

kchouse_m_bedrooms = kchouse.groupby('bedrooms', as_index=True).median()

kchouse_m_bedrooms.head()
#寝室の数と価格のグラフ

plt.figure(figsize=(20,10))

plt.title("Total Bedrooms & median Price")

sns.barplot(x=kchouse_m_bedrooms.index, y="price",data=kchouse_m_bedrooms,palette="viridis")
#バスルームの数を基軸に中央値でグルーピング

kchouse_m_bathrooms=kchouse.groupby("bathrooms",as_index=True).median()



#バスルームの数と価格のグラフ

plt.figure(figsize=(20,10))

plt.title("Total bathrooms & median Price")

sns.barplot(x=kchouse_m_bathrooms.index, y="price",data=kchouse_m_bathrooms,palette="viridis")
#bathroomsの数が7以上

kchouse[kchouse["bathrooms"]>6]
#物件面積(sqft_living)

sns.jointplot(x="price",y="sqft_living", data=kchouse, kind="reg", size=10, color="midnightblue")
#物件階数（floors)を基軸に中央値でグルーピング

kchouse_m_floors=kchouse.groupby("floors",as_index=True).median()



#バスルームの数と価格のグラフ

plt.figure(figsize=(20,10))

plt.title("Total floors & median Price")

sns.barplot(x=kchouse_m_floors.index, y="price",data=kchouse_m_floors,palette="viridis")
#湖岸景色有無（waterfront)

#湖岸景色有無（waterfront)を基軸に中央値でグルーピング

kchouse_m_waterfront=kchouse.groupby("waterfront",as_index=True).median()



#湖岸景色有無（waterfront)と価格のグラフ

plt.figure(figsize=(20,10))

plt.title("waterfront & median Price")

sns.barplot(x=kchouse_m_waterfront.index, y="price",data=kchouse_m_waterfront,palette="viridis")
#内件された数（view)

#内件された数（views)を基軸に中央値でグルーピング

kchouse_m_view=kchouse.groupby("view",as_index=True).median()



#内件された数（view)と価格のグラフ

plt.figure(figsize=(20,10))

plt.title("view & median Price")

sns.barplot(x=kchouse_m_view.index, y="price",data=kchouse_m_view,palette="viridis")
#物件のコンディション(condition)

#物件のコンディション(condition)を基軸に中央値でグルーピング

kchouse_m_condition=kchouse.groupby("condition",as_index=True).median()



#物件のコンディション(condition)と価格のグラフ

plt.figure(figsize=(20,10))

plt.title("condition & median Price")

sns.barplot(x=kchouse_m_condition.index, y="price",data=kchouse_m_condition,palette="viridis")
#物件のグレード(grade)

#物件のグレード(grade)を基軸に中央値でグルーピング

kchouse_m_grade=kchouse.groupby("grade",as_index=True).median()



#物件のグレード(grade)と価格のグラフ

plt.figure(figsize=(20,10))

plt.title("grade & median Price")

sns.barplot(x=kchouse_m_grade.index, y="price",data=kchouse_m_grade,palette="viridis")
#地上部広さ(sqft_above)

sns.jointplot(x="price",y="sqft_above",data=kchouse,kind="reg",size=10,color="midnightblue")
#地下室広さ(sqft_basement)

sns.jointplot(x="price",y="sqft_basement",data=kchouse,kind="reg",size=10,color="midnightblue")
#地下室がない物件の割合を計算

float(kchouse["sqft_basement"][kchouse["sqft_basement"]==0].count())/21613*100
#建造年（yr_built)

#建造年（yr_built)を基軸に中央値でグルーピング

kchouse_m_yr_built=kchouse.groupby("yr_built",as_index=True).median()



#建造年（yr_built)と価格のグラフ

plt.figure(figsize=(20,10))

plt.title("yr_built & median Price")

sns.barplot(x=kchouse_m_yr_built.index, y="price",data=kchouse_m_yr_built,palette="viridis")
#リノベーション（yr_renovated)を基軸に中央値でグルーピング

kchouse_m_yr_renovated=kchouse.groupby("yr_renovated",as_index=True).median()



#リノベーション（yr_renovated)と価格のグラフ

plt.figure(figsize=(20,10))

plt.title("yr_renovated & median Price")

sns.barplot(x=kchouse_m_yr_renovated.index, y="price",data=kchouse_m_yr_renovated,palette="viridis")



float(kchouse["yr_renovated"][kchouse["yr_renovated"]==0].count())/21613*100
#案１ リフォームフラグの処理

kchouse["renovatedflag"]=np.where(kchouse["yr_renovated"]==0, 0, 1)



#リフォームフラグと価格中央値の可視化

ren_flag=kchouse.groupby("renovatedflag",as_index=True).median()

plt.figure(figsize=(10,10))

sns.barplot(x=ren_flag.index, y="price", data=ren_flag, palette="viridis")
#案２　物件面積の変化量の処理

kchouse["sqft_living_Chg"]=kchouse["sqft_living"]-kchouse["sqft_living15"]



#物件面積の変化量と価格の可視化

sns.jointplot(x="price",y="sqft_living_Chg", data=kchouse, kind="reg", size=10, color="midnightblue")

#案３　駐車場面積の変化量

kchouse["Sqft_lot_Chg"]=kchouse["sqft_lot"]-kchouse["sqft_lot15"]



#駐車場面積の変化量と価格の可視化

sns.jointplot(x="price",y="Sqft_lot_Chg", data=kchouse, kind="reg", size=10, color="midnightblue")
#データセット確認

kchouse.head()
#訓練データとテストデータの切り分け

train_set, test_set=train_test_split(kchouse,test_size=0.2,random_state=42)
#訓練データ切り分け

X_train=train_set.drop("price", axis=1)

y_train=train_set["price"].copy()



X_test=test_set.drop("price", axis=1)

y_test=test_set["price"].copy()
#ランダムフォレストレグレッサー

RFclf=RandomForestRegressor(n_estimators=100, criterion="mse")



#モデル訓練

RFclf=RFclf.fit(X_train,y_train)
#各特徴量を重要度順にソートして表示

features=X_train.columns

importances=RFclf.feature_importances_



print("Features sorted by most importance:")

print(sorted(zip(map(lambda x: round(x,2), RFclf.feature_importances_), features),reverse=True))
#重要度の高い説明変数（上位５位）を切りわける

X_train=X_train[["grade", "sqft_living", "lat", "long", "waterfront"]]

X_test=X_test[["grade", "sqft_living", "lat", "long", "waterfront"]]
#3層の決定木モデル

clf=DecisionTreeRegressor(max_depth=3)



#モデルの訓練

clf=clf.fit(X_train,y_train)



#訓練データを使って予測

y_pred=clf.predict(X_train)

#RMSE計算（訓練データ使用）

np.sqrt(mean_squared_error(y_train,y_pred))
#テストデータを使って予測

y_pred_test=clf.predict(X_test)



#RMSE計算（テストデータ使用）

np.sqrt(mean_squared_error(y_test,y_pred_test))
#決定木の視覚化

dot_data=tree.export_graphviz(clf,out_file=None,

                             feature_names=X_train.columns,

                             class_names=X_train, 

                             filled=True,

                             rounded=True,

                             special_characters=True)

graph=graphviz.Source(dot_data)

graph
#決定木１００本のランダムフォレスト

RFclf=RandomForestRegressor(n_estimators=100)



#訓練データのトレーニング

RFclf=RFclf.fit(X_train,y_train)



#訓練データの予測

RF_y_pred=RFclf.predict(X_train)
#RMSE計算（訓練データ使用）

np.sqrt(mean_squared_error(y_train,RF_y_pred))
#テストデータの予測

RF_y_pred_test=RFclf.predict(X_test)
#RMSE計算（テストデータ使用）

np.sqrt(mean_squared_error(y_test,RF_y_pred_test))
#実施値と予測値のデータフレーム

DFRFtest=pd.DataFrame({"Actual":y_test, "Prediction":RF_y_pred_test})



#データ確認

DFRFtest.head()