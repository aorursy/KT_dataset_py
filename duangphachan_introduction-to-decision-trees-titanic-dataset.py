!pip install pydotplus
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (roc_curve, auc, accuracy_score)

#可視化用

import pydotplus

from IPython.display import Image

from graphviz import Digraph

from sklearn.externals.six import StringIO

from sklearn import tree





train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')


#欠損値処理

train['Fare'] = train['Fare'].fillna(train['Fare'].median())

train['Age'] = train['Age'].fillna(train['Age'].median())

train['Embarked'] = train['Embarked'].fillna('S')



#カテゴリ変数の変換

train['Sex'] = train['Sex'].apply(lambda x: 1 if x == 'male' else 0)

train['Embarked'] = train['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)



train = train.drop(['Cabin','Name','PassengerId','Ticket'],axis=1)

# train = train.drop(['Cabin','Name','PassengerId','Ticket', 'SibSp', 'Parch', 'Embarked', 'Fare'],axis=1)

train_X = train.drop('Survived', axis=1)

train_y = train.Survived

(train_X, test_X ,train_y, test_y) = train_test_split(train_X, train_y, test_size = 0.3, random_state = 666)
train_X
# Train

# max_depthの値を変えてみよう（１〜１０）、Accuracyの結果が変わることを確認

clf = DecisionTreeClassifier(random_state=0, max_depth=6)

clf = clf.fit(train_X, train_y)



# Predict

pred = clf.predict(test_X)

fpr, tpr, thresholds = roc_curve(test_y, pred, pos_label=1)

auc(fpr, tpr)

accuracy_score(pred, test_y)


dot_data = StringIO()

tree.export_graphviz(clf, out_file=dot_data,feature_names=train_X.columns, max_depth=3)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

graph.write_pdf("graph.pdf")

Image(graph.create_png())
# 提出用のデータを予測し、CSVファイルとして出力

test
test_features = test[["Pclass", "Sex", "Age", "Fare"]].values
# test_without_PId = tree_data["test"].drop("PassengerId",axis=1)

clf.predict(test_features)
#欠損値処理

test['Fare'] = test['Fare'].fillna(test['Fare'].median())

test['Age'] = test['Age'].fillna(test['Age'].median())

test['Embarked'] = test['Embarked'].fillna('S')



#カテゴリ変数の変換

test['Sex'] = test['Sex'].apply(lambda x: 1 if x == 'male' else 0)

test['Embarked'] = test['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)



test_features = test.drop(['Cabin','Name','PassengerId','Ticket'],axis=1)
pred_value = clf.predict(test_features).astype(int)
test['PassengerId']
pred_result = pd.DataFrame({"PassengerId":test['PassengerId'],

                           "Survived":pred_value})
pred_result.to_csv("lt_decesion_tree_predict_result.csv", index=False)