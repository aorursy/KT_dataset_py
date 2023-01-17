import pandas as pd

import numpy as np



train = pd.read_csv("/kaggle/input/titanic/train.csv")

test = pd.read_csv("/kaggle/input/titanic/test.csv")
#データのサイズを確認

test_shape = test.shape

train_shape = train.shape



print(test_shape)

print(train_shape)
test.describe()
train.describe()
#　欠損データを代理データに入れ替える

train["Age"] = train["Age"].fillna(train["Age"].median())

train["Embarked"] = train["Embarked"].fillna("S")
#文字を数字に変換

train["Sex"][train["Sex"] == "male"] = 0

train["Sex"][train["Sex"] == "female"] = 1

train["Embarked"][train["Embarked"] == "S" ] = 0

train["Embarked"][train["Embarked"] == "C" ] = 1

train["Embarked"][train["Embarked"] == "Q"] = 2



#表示して変換されているか確認

train.head(10)
#文字を数字に変換

test["Age"] = test["Age"].fillna(test["Age"].median())

test["Sex"][test["Sex"] == "male"] = 0

test["Sex"][test["Sex"] == "female"] = 1

test["Embarked"][test["Embarked"] == "S"] = 0

test["Embarked"][test["Embarked"] == "C"] = 1

test["Embarked"][test["Embarked"] == "Q"] = 2

test.Fare[152] = test.Fare.median()



#表示して変換されているか確認

test.head(10)
#scikitlearnのインポート

from sklearn import tree



# 「train」の目的変数と説明変数の値を取得

target = train["Survived"].values

features_one = train[["Pclass", "Sex", "Age", "Fare"]].values

# 決定木の作成

kettei_tree = tree.DecisionTreeClassifier()

kettei_tree = kettei_tree.fit(features_one, target)

# 「test」の説明変数の値を取得

test_features = test[["Pclass", "Sex", "Age", "Fare"]].values

# 「test」の説明変数を使って「kettei_tree」のモデルで予測

prediction = kettei_tree.predict(test_features)
# 予測データのサイズを確認

prediction.shape
#予測データの中身を確認

print(prediction)
# PassengerIdを取得

PassengerId = np.array(test["PassengerId"]).astype(int)

# prediction(予測データ）とPassengerIdをデータフレームへ落とし込む

solution = pd.DataFrame(prediction, PassengerId, columns = ["Survived"])

# kettei_tree.csvとして書き出し

solution.to_csv("kettei_tree.csv", index_label = ["PassengerId"])