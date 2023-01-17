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
from sklearn import tree
 
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

print(train)

test_shape = test.shape
train_shape = train.shape

print(test_shape)
print(train_shape)

test.describe()
train.describe()

print(train.describe())
print(test.describe())

def kesson_table(df): 
        null_val = df.isnull().sum()
        percent = 100 * df.isnull().sum()/len(df)
        kesson_table = pd.concat([null_val, percent], axis=1)
        kesson_table_ren_columns = kesson_table.rename(
        columns = {0 : '欠損数', 1 : '%'})
        return kesson_table_ren_columns
 
kesson_table(train)
kesson_table(test)

print(kesson_table(train))
print(kesson_table(test))

def Gorira(aho):
    print(aho,"ごりらです。")
    
Gorira("アホ")
Gorira("アンポンタン")

train["Age"] = train["Age"].fillna(train["Age"].median())
train["Embarked"] = train["Embarked"].fillna("S")
 
kesson_table(train)

train

train.Embarked = train.Embarked.replace(['C', 'S', 'Q'], [0, 1, 2])
train.Sex = train.Sex.replace(['male', 'female'], [0, 1])
 
print(train.head(10))

test["Age"] = test["Age"].fillna(test["Age"].median())
test.Embarked = test.Embarked.replace(['C', 'S', 'Q'], [0, 1, 2])
test.Sex = test.Sex.replace(['male', 'female'], [0, 1])
test.Fare[152] = test.Fare.median()

print(test.head(10))

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


# 予測データのサイズを確認
my_prediction.shape

#予測データの中身を確認
print(my_prediction)

# PassengerIdを取得
PassengerId = np.array(test["PassengerId"]).astype(int)
 
# my_prediction(予測データ）とPassengerIdをデータフレームへ落とし込む
my_solution = pd.DataFrame(my_prediction, PassengerId, columns = ["Survived"])
 
# my_tree_one.csvとして書き出し
my_solution.to_csv("my_tree_one.csv", index_label = ["PassengerId"])
 




  
