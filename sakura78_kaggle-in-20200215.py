# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# ライブラリを読込

import numpy as np

import pandas as pd

 

# CSVを読み込んでデータをセット

train = pd.read_csv("../input/titanic/train.csv")

test = pd.read_csv("../input/titanic/test.csv")



# 全体情報を出力する

test.info()

train.info()



# 先頭行を出力する

test.head(3)

train.head(3)



# ★train の乗客の情報と「Survived（生存したかどうか）」の答えを機械学習して、

# 　test で提供されている乗客情報を元に、生存したか死亡したかの予測を作るのが課題
#行数、列数、全要素数（サイズ）を取得

test_shape = test.shape

train_shape = train.shape

 

print(test_shape)

print(train_shape)

 
# 各データセットの基本統計量を確認

#test.describe()

train.describe()
# カラムに入っている情報の一覧

train["Parch"].unique()

train["SibSp"].unique()

# train["Age"].unique()

# train["Fare"].unique()
def kesson_table(df): 

        null_val = df.isnull().sum()

        percent = 100 * df.isnull().sum()/len(df)

        kesson_table = pd.concat([null_val, percent], axis=1)

        kesson_table_ren_columns = kesson_table.rename(

        columns = {0 : '欠損数', 1 : '%'})

        return kesson_table_ren_columns

 

# kesson_table(train)

kesson_table(test)
# 欠損データを補間

# AgeとFareは数値なので中央値を代入

train["Age"] = train["Age"].fillna(train["Age"].median())

test["Age"] = test["Age"].fillna(test["Age"].median())



train["Fare"] = train["Fare"].fillna(train["Fare"].median())

test["Fare"] = test["Fare"].fillna(test["Fare"].median())



# Enbarkedは文字列なので一番多い"S"を代入

train["Embarked"] = train["Embarked"].fillna("S")

test["Embarked"] = test["Embarked"].fillna("S")



kesson_table(test)
# 文字列データを数値に変換

# train["Sex"][train["Sex"] == "male"] = 0

# train["Sex"][train["Sex"] == "female"] = 1



# カテゴリ変数をダミー変数に変換

# 男女と出港地はカテゴリであり数値ではないので、カラム要素の種類数に

# 関わらず、ダミー変数に分割してあげる

train = pd.get_dummies(train, columns=["Sex","Embarked"])

test = pd.get_dummies(test, columns=["Sex","Embarked"])



#train["Embarked"][train["Embarked"] == "S" ] = 0

#train["Embarked"][train["Embarked"] == "C" ] = 1

#train["Embarked"][train["Embarked"] == "Q"] = 2



#train.head(3)

test.head(3)
# scikit-learnのインポートをします

from sklearn import tree
# 「train」の目的変数と説明変数の値を取得

target = train["Survived"].values

features_one = train[["Pclass", "Sex_female", "Sex_male", "Age", "Fare"]].values

 

# 決定木の作成

my_tree_one = tree.DecisionTreeClassifier()

my_tree_one = my_tree_one.fit(features_one, target)

 

# 「test」の説明変数の値を取得

test_features = test[["Pclass", "Sex_female", "Sex_male", "Age", "Fare"]].values

 

# 「test」の説明変数を使って「my_tree_one」のモデルで予測

my_prediction = my_tree_one.predict(test_features)

 
# 予測データのサイズを確認

my_prediction.shape

 
#予測データの中身を確認

print(my_prediction)
# PassengerIdを取得

PassengerId = np.array(test["PassengerId"]).astype(int)

 

# my_prediction(予測データ）とPassengerIdをデータフレームへ落とし込む

my_solution = pd.DataFrame(my_prediction, PassengerId, columns = ["Survived"])

 

# my_tree_one.csvとして書き出し

my_solution.to_csv("my_tree_one2.csv", index_label = ["PassengerId"])

 

 