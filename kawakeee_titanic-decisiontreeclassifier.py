import pandas as pd

import numpy as np
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")



print("train.shapeの大きさ:",train.shape)

print("test.shapeの大きさ:",test.shape)
#12の特徴量が存在する事を確認

train.head()
#11の特徴量が存在する事を確認

test.head()
#pandasのdescribe()関数を読んで基本統計量を確認

train.describe()

#countの行を確認

#Ageが未設定の行数が　（891-714）＝177　行ある
test.describe()

#countの行を確認

#Ageが未設定の行数が　（418-332）＝８６　行ある

#Fareが未設定の行数が　（418-417）＝1　行ある
# ①欠損データの確認

def kesson_table(df): 

        null_val = df.isnull().sum()

        percent = 100 * df.isnull().sum()/len(df)

        kesson_table = pd.concat([null_val, percent], axis=1)

        kesson_table_ren_columns = kesson_table.rename(

        columns = {0 : '欠損数', 1 : '%'})

        return kesson_table_ren_columns



print("訓練データの欠損情報")

kesson_table(train)
print("テストデータの欠損情報")

kesson_table(test)
# ②欠損データの事前処理

# ②-(1) 欠損データを代理データに入れ替える

# 「Cabin」は予測モデルで使わないので、「Age」と「Embarked」の2つの欠損データを補完する



#訓練データのAgeの欠損箇所に、訓練データのAgeの中央値を代入する

# pandas.DataFrame.fillna() 欠損値を引数の値に置き換える

train["Age"] = train["Age"].fillna(train["Age"].median())

#訓練データのEmbarkedの欠損箇所に、Sを代入する

train["Embarked"] = train["Embarked"].fillna("S")



#訓練データにて欠損がなくなった事を確認(Cabinは除く)

kesson_table(train)
# ②-(2) 文字列カテゴリ列データを数字へ変換

# 予想で使う項目で文字列を値として持っているカラムは「Sex」と「Embarked」

# Sexは「male」「female」の２つの文字列値

# Embarkedはは「S」「C」「Q」の3つの文字列値　これらを数字に変換する。



# Sexにてmaleを0 femaleを1　に変換

train["Sex"][train["Sex"] == "male"] = 0

train["Sex"][train["Sex"] == "female"] = 1



# EmbarkedにてSを0 Cを1　Qを2 に変換

train["Embarked"][train["Embarked"] == "S" ] = 0

train["Embarked"][train["Embarked"] == "C" ] = 1

train["Embarked"][train["Embarked"] == "Q"] = 2



# 最初の10行を見て変換されたかを確認

train.head(10)
#テストデータにおいても同様に、欠損データの事前処理を行う

test["Age"] = test["Age"].fillna(test["Age"].median())

test["Sex"][test["Sex"] == "male"] = 0

test["Sex"][test["Sex"] == "female"] = 1

test["Embarked"][test["Embarked"] == "S"] = 0

test["Embarked"][test["Embarked"] == "C"] = 1

test["Embarked"][test["Embarked"] == "Q"] = 2

#テストデータではFareが一つ欠損しているので、中央値を設定する

test.Fare[152] = test.Fare.median()



# 最初の10行を見て変換されたかを確認

test.head(10)
# ③予測モデル（決定木）を構築する

# scikit-learnからtreeをインポートをする

from sklearn import tree

# 説明変数としては"Pclass","Age","Sex","Fare", "SibSp", "Parch", "Embarked"を使用する

features_col = ["Pclass","Age","Sex","Fare", "SibSp", "Parch", "Embarked"]

features = train[features_col].values

# 目的変数として"Survived"を取得

target = train["Survived"].values
from sklearn.model_selection import GridSearchCV

#グリッドサーチの範囲を指定

parameters = {

    "max_depth":[i for i in range(1,15,1)],

    "min_samples_split": [2,4,5,10,12,16],

}

#交差検証+グリッドサーチにより最良パラメータの検索

clf = GridSearchCV(tree.DecisionTreeClassifier(), parameters, cv=5)

clf.fit(features, target)
# 最良パラメータ: {'max_depth': 6, 'min_samples_split': 12} 

# 最良交差検証スコア: 0.82

# public score:0.77511

print("最良パラメータ: {}".format(clf.best_params_))

print("最良交差検証スコア: {:.2f}".format(clf.best_score_))
# 最良パラメータで改めて決定木モデルの作成

# 最良パラメータ: {'max_depth': 6, 'min_samples_split': 12} 

# public score:0.77511

my_tree = tree.DecisionTreeClassifier(max_depth=6, min_samples_split=12)

my_tree.fit(features, target)
# 「test」の説明変数の値を取得

test_features = test[features_col].values

# 「test」の説明変数を使って「clf」のモデルで予測

my_prediction = my_tree.predict(test_features)



# 予測データのサイズを確認

print("my_predictionの大きさ：",my_prediction.shape)

#予測データの中身を確認

print(my_prediction)

# ④予測値を取得して提出用CSVファイルを作成

# 元のテストデータからPassengerIdを取得

PassengerId = np.array(test["PassengerId"]).astype(int)

# my_prediction(予測データ）とPassengerIdをデータフレームへ落とし込む

my_submission = pd.DataFrame(my_prediction, PassengerId, columns = ["Survived"])

# my_tree_one.csvとして書き出し　インデックスラベルとしてPassengerIdを指定

my_submission.to_csv("submission.csv", index_label = ["PassengerId"])

print(my_submission.shape)
my_submission.head()
features.shape
from sklearn.tree import export_graphviz

# 決定木の可視化ファイルの生成

export_graphviz(my_tree, out_file="my_tree.dot", class_names=["alive", "dead"],

                feature_names=features_col, impurity=False, filled=True)

# 全体を確認

from IPython.display import Image, display_png

!dot -Tpng my_tree.dot -o my_tree.png

display_png(Image("my_tree.png"))



# 拡大表示

import graphviz

with open("my_tree.dot") as f:

    dot_graph = f.read()

display(graphviz.Source(dot_graph))