# ライブラリ読み込み

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import seaborn as sns

%matplotlib inline



import sklearn

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestClassifier



sns.set(style='white', context='notebook', palette='deep')
# データ読み込み

df_train = pd.read_csv("/kaggle/input/titanic/train.csv")

df_test = pd.read_csv("/kaggle/input/titanic/test.csv")



df_train.head()
df_test_id = df_test["PassengerId"]

df_train = df_train[["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked","Survived"]]

df_test = df_test[["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]]



df_train.head()
df_train.info()
df_test.info()
# Ageの欠損値を平均値で埋める

age_ave = df_train['Age'].mean()

df_train['Age'] = df_train['Age'].fillna(age_ave)

df_test['Age'] = df_test['Age'].fillna(age_ave)



# Fareの欠損値を平均値で埋める

fare_ave = df_train['Fare'].mean()

df_train['Fare'] = df_train['Fare'].fillna(fare_ave)

df_test['Fare'] = df_test['Fare'].fillna(fare_ave)



# Embarkedの欠損値を最頻値"S"で埋める

df_train['Embarked'] = df_train['Embarked'].fillna("S")

df_test['Embarked'] = df_test['Embarked'].fillna("S")
# SibSpとParchの和が1以上であれば1、そうでなければ0となる列"hasFamily"を追加

df_train["FamilyNum"] = df_train["SibSp"] + df_train["Parch"]

df_train["hasFamily"] = df_train["FamilyNum"].apply(lambda x : 1 if x >= 1 else 0)

df_train = df_train.drop(labels = ["SibSp"], axis = 1)

df_train = df_train.drop(labels = ["Parch"], axis = 1)



df_test["FamilyNum"] = df_test["SibSp"] + df_test["Parch"]

df_test["hasFamily"] = df_test["FamilyNum"].apply(lambda x : 1 if x >= 1 else 0)

df_test = df_test.drop(labels = ["SibSp"], axis = 1)

df_test = df_test.drop(labels = ["Parch"], axis = 1)
# カテゴリー変数をダミー変数にエンコード

df_train = pd.get_dummies(df_train, columns=["Sex", "Pclass", "Embarked"])

df_test = pd.get_dummies(df_test, columns=["Sex", "Pclass", "Embarked"])



# エンコード後のデータ構造を確認するため表示

df_train.head()
# 特徴量とラベルに分割

Y_train = df_train["Survived"]

X_train = df_train.drop(labels = ["Survived"], axis = 1)

X_test = df_test



del df_train

del df_test
# 学習用セットと検証用セットに分割

random_seed = 1

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.5, random_state = random_seed)
# 学習に関するパラメータ設定

params = {

    "n_estimators" : [2, 5, 10, 15, 20, 30, 50, 75, 100, 200, 500, 1000],

    "criterion" : ["gini"],

    "min_samples_split" : [2, 3, 5, 10, 15, 20, 30],

    "max_depth" : [2, 3, 5, 10, 15, 20, 30],

    "random_state" : [1],

    "verbose" : [False],

}



# モデル構築

model = GridSearchCV(RandomForestClassifier(), params, cv = 3)

model = model.fit(X_train, Y_train)

model = model.best_estimator_



# 検証用セットを用いて評価

model.score(X_val, Y_val)
# 予測

results = model.predict(X_test)

results = pd.Series(results, name = "Survived")
# 提出データを作成

submission = pd.concat([df_test_id, results], axis = 1)

submission.to_csv("titanic_submission.csv", index = False)

submission
features = X_train.columns

importance = model.feature_importances_

indices = np.argsort(importance)



plt.figure(figsize = (8,6))

plt.barh(range(len(indices)), importance[indices], color='b', align='center')

plt.yticks(range(len(indices)), features[indices])

plt.show()