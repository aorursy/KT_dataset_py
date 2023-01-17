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
#訓練データとテストデータの読み込み

train= pd.read_csv("../input/train.csv")

test= pd.read_csv("../input/test.csv")

#先頭の５データの表示

train.head()
#訓練データの情報を表示

train.info()
#テストデータの情報の表示

test.info()
#訓練データの基本統計量の確認

train.describe()
#テストデータの基本統計量の確認

test.describe()
#訓練データのAgeの欠損値を中央値で埋める

train["Age"] = train["Age"].fillna(train["Age"].median())

#訓練データのEmbarkedをSで埋める

train["Embarked"] = train["Embarked"].fillna("S")
#カテゴリカルデータの数値への置き換え

train["Sex"][train["Sex"] == "male"] = 0

train["Sex"][train["Sex"] == "female"] = 1

train["Embarked"][train["Embarked"] == "S" ] = 0

train["Embarked"][train["Embarked"] == "C" ] = 1

train["Embarked"][train["Embarked"] == "Q"] = 2
train.head()
#テストデータも同様に処理

test["Age"] = test["Age"].fillna(test["Age"].median())

test["Sex"][test["Sex"] == "male"] = 0

test["Sex"][test["Sex"] == "female"] = 1

test["Embarked"][test["Embarked"] == "S"] = 0

test["Embarked"][test["Embarked"] == "C"] = 1

test["Embarked"][test["Embarked"] == "Q"] = 2

test.Fare[152] = test.Fare.median()
# scikit-learnの決定木をインポートをします

from sklearn import tree
# 「train」の目的変数と説明変数の値を取得

target = train["Survived"].values

features_one = train[["Pclass", "Sex", "Age", "Fare"]].values

 

# 決定木の作成

my_tree_one = tree.DecisionTreeClassifier()

my_tree_one = my_tree_one.fit(features_one, target)

 

# 「test」の説明変数の値を取得

test_features = test[["Pclass", "Sex", "Age", "Fare"]].values

 

# 「test」の説明変数を使って「my_tree_one」のモデルで予測

my_prediction = my_tree_one.predict(test_features)
#予測データの中身を確認

print(my_prediction)
# PassengerIdを取得

PassengerId = np.array(test["PassengerId"]).astype(int)

 

# my_prediction(予測データ）とPassengerIdをデータフレームへ落とし込む

my_solution = pd.DataFrame(my_prediction, PassengerId, columns = ["Survived"])

 

# my_tree_one.csvとして書き出し

my_solution.to_csv("my_tree_one.csv", index_label = ["PassengerId"])