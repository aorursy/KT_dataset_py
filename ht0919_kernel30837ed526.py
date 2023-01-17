from sklearn.ensemble import GradientBoostingClassifier

import pandas as pd

import numpy as np



train = pd.read_csv("/kaggle/input/titanic/train.csv")

test = pd.read_csv("/kaggle/input/titanic/test.csv")



# 前処理：欠損データの補完

# 欠損データを中央値や最頻値で置き換える

train["Age"] = train["Age"].fillna(train["Age"].median())

train["Embarked"] = train["Embarked"].fillna("S")

# 前処理：カテゴリカルデータの数値化

# Sexは['male','female']を[0,1]に、Embarkedは['S','C','Q']を[0,1,2]に変換

train["Sex"] = train.Sex.replace("male",0).replace("female",1)

train["Embarked"] = train.Embarked.replace("S",0).replace("C",1).replace("Q",2)

# testもtrainと同様に前処理

test["Age"] = test["Age"].fillna(test["Age"].median())

test["Fare"] = test["Fare"].fillna(test["Fare"].median())

test["Sex"] = test.Sex.replace("male",0).replace("female",1)

test["Embarked"] = test.Embarked.replace("S",0).replace("C",1).replace("Q",2)



#　家族数を追加

train_two = train.copy()

train_two["family_size"] = train_two["SibSp"] + train_two["Parch"] + 1

test_two = test.copy()

test_two["family_size"] = test_two["SibSp"] + test_two["Parch"] + 1



# 「train」の目的変数と説明変数の値を取得

target = train["Survived"].values

features = train_two[["Pclass", "Age", "Sex", "Fare", "family_size", "Embarked"]].values



# モデルは勾配ブースティング

forest = GradientBoostingClassifier(n_estimators=55, random_state=9)

forest = forest.fit(features, target)



# testから値を取り出す

test_features = test_two[["Pclass", "Age", "Sex", "Fare", "family_size", "Embarked"]].values



# 予測をしてCSVへ書き出す

my_prediction_forest = forest.predict(test_features)

PassengerId = np.array(test["PassengerId"]).astype(int)

my_solution_forest = pd.DataFrame(my_prediction_forest, PassengerId, columns=["Survived"])

my_solution_forest.to_csv("/kaggle/working/gender_submission.csv", index_label=["PassengerId"])