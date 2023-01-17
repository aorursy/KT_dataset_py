import numpy as np

import pandas as pd

import seaborn as sns

import os



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv("../input/titanic/train.csv")

test = pd.read_csv("../input/titanic/test.csv")
# まず最初に、トレーニングデータとテストデータの行数、列数を確認しましょう

# 1-1) トレーニングデータの先頭5件を表示してみましょう

# 1-2) テストデータの先頭5件を表示してみましょう 

# 2-1) infoメソッドでトレーニングデータの情報を表示してみましょう

# 2-2) トレーニングデータの欠損値状況を確認しましょう

# 2-3) テストデータの欠損値状況を確認しましょう

# 2-4) ターゲット変数 Survived の値毎の件数を確認してみましょう

# 2-5) 変数 Pclass にはどんな値が設定されているか確認してみましょう

# 2-6) 変数 Age の分布をヒストグラムで確認してみましょう

# 2-7) 変数 Age の最大値、平均値、中央値を確認してみましょう

# 2-8) 変数 Sex の分布をvalue_counts＋棒グラフで確認してみましょう

# 2-9) pd.crosstabを使って、【Survived毎】の 変数 Sex の件数を確認してみましょう

# 3-1) 【Survived毎】の 変数 Sex の件数を棒グラフで確認してみましょう

# 相関関係はありそうですか？あるとしたらどのような傾向がありますか？

# 3-2) 【Survived毎】の 変数 Pclass の件数を棒グラフで確認してみましょう

# 相関関係はありそうですか？あるとしたらどのような傾向がありますか？

# 欠損値を埋める

train['Age'].fillna(0, inplace=True)

test['Age'].fillna(0, inplace=True)
from sklearn.preprocessing import LabelEncoder



'''

トレーニングデータとテストデータが同じ値の分布であれば、直接One-Hotエンコーディングでも構いません。  

異なる値の分布であれば、共通のLabelEncoderを使用して数値化しましょう。  

他のNotebookではトレーニングデータとテストデータを結合したものにOne-Hotエンコーディングを適用しているものもあります。

'''



_='''

for f in ['XXX']:

    le = LabelEncoder()

    le.fit(pd.concat([train[f], test[f]]))

    train[f] = le.transform(train[f])

    test[f] = le.transform(test[f])

'''
# 文字列項目 --> One-Hotエンコーディング

train = pd.get_dummies(train, columns=['Sex','Embarked'])

test = pd.get_dummies(test, columns=['Sex','Embarked'])
# 不要な列の削除

drop_columns = ['PassengerId','Name','Ticket','Cabin','Parch','SibSp','Fare','Age']



train.drop(drop_columns, axis=1, inplace=True)

test.drop(drop_columns, axis=1, inplace=True)
# 編集後のデータを確認

train.head()
# 欠損値がないことを確認

print(train.isna().any())

print()

print(test.isna().any())
X_train = train.drop(['Survived'], axis=1)

Y_train = train['Survived']

X_test = test
from sklearn.dummy import DummyClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn import tree

import graphviz
_='''

'''

# モデル構築（ランダム分類、50%/50%）

clf = DummyClassifier(strategy='uniform', random_state=42)



# 学習

clf.fit(X_train, Y_train)



# 予測

result = clf.predict(X_test)



# 結果表示

print(pd.Series(result).value_counts())
_='''

# モデル構築（決定木）

clf = DecisionTreeClassifier(max_leaf_nodes=6)



# 学習

clf.fit(X_train, Y_train)



# 予測

result = clf.predict(X_test)



# 結果表示

print(pd.Series(result).value_counts())



# 結果表示 - 決定木の構造表示

tree_graph = tree.export_graphviz(clf, out_file=None, max_depth=10,

    impurity=False, feature_names=X_train.columns, class_names=['0', '1'],

    rounded=True, filled=True )

graphviz.Source(tree_graph)

'''
_='''

# モデル構築（ランダムフォレスト）

clf = RandomForestClassifier(n_estimators=10, random_state=42)



clf.fit(X_train, Y_train)



result = clf.predict(X_test)



pd.Series(result).value_counts()

'''
_='''

'''

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score



# X_trainとY_trainをtrainとvalidに分割

train_x, valid_x, train_y, valid_y = train_test_split(X_train, Y_train, test_size=0.2, random_state=0)



''' 前の処理で宣言したclfのモデルを使います '''

clf.fit(train_x, train_y)



# valid_xについて推論

oof = clf.predict(valid_x)



# 正解率を表示

print('score', round(accuracy_score(valid_y, oof), 3)) 
# テストデータの PassengerId を捨てていたので読み込み直す

test = pd.read_csv("../input/titanic/test.csv")



# 指定フォーマットのサブミットファイル作成

submission = pd.DataFrame({

    "PassengerId": test["PassengerId"],

    "Survived": result

})

submission.to_csv("submission.csv", index=False)