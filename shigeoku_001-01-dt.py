import numpy as np
import pandas as pd

from sklearn import tree
from sklearn.metrics import  confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

import graphviz
# pickle ファイルから読み込み
train_pkl = pd.read_pickle('./pd_train.pk2')
train_pkl.shape
# Age, Fare は削除
# ダミー変数の先頭は削除
train_pkl.drop(['Age', 'Fare', 'Sex_female', 'Embarked_C', 'Pclass_2'], inplace=True, axis=1)
train_pkl.describe()
# 訓練とテストデータに分割
train, test = train_test_split(train_pkl, test_size=0.2, random_state=42)
# ターゲットと特徴量の分割
train_X = train.iloc[:, 1:].values
train_y = train.Survived.values
DT = tree.DecisionTreeClassifier(random_state=42)
DT = DT.fit(train_X, train_y)
train.Survived.name
len(train.iloc[:, 1:].columns), train.iloc[:, 1:].columns
# 訓練済みの決定木を視覚化
dot_data = tree.export_graphviz(DT, out_file=None,
                               feature_names=train.iloc[:, 1:].columns,
                               class_names=train.Survived.name,
                               rounded=True,
                               filled=True,
                               special_characters=True)
graph = graphviz.Source(dot_data)
graph
# graph.write('.\DT.png')
type(train_X)
DT.feature_importances_
# 特徴量の重要度が高い順に表示
print("特徴量の重要度が高い順：")
# sorted：reverse=True 降順
print(sorted(
    zip(map(lambda x: round(x, 3), DT.feature_importances_), train.iloc[:, 1:].columns),
    reverse=True))
sorted(
    zip(map(lambda x: round(x, 3), DT.feature_importances_), train.iloc[:, 1:].columns),
    reverse=True)
# ターゲットと特徴量の分割
test_x = test.iloc[:, 1:].values
test_y = test.Survived.values
test_x.shape, test_y.shape
pred_y = DT.predict(test_x)
confusion_matrix(test_y, pred_y)
accuracy_score(test_y, pred_y)
# 検証データ読み込み
valid = pd.read_pickle('./pd_test.pk2')
valid.shape
# ID の保存
valid_pass = valid.PassengerId.values
valid_X = valid.iloc[:, 1:]
valid_X.describe()
valid_X.drop(['Age', 'Fare', 'Sex_female', 'Embarked_C', 'Pclass_2'], inplace=True, axis=1)
valid_X.shape, train_X.shape
pred_valid_y = DT.predict(valid_X)
pred_valid_y.shape
type(valid_pass), type(pred_valid_y)
result_df = pd.DataFrame(pred_valid_y, valid_pass, columns=['Survived'])
result_df.to_csv("./tree_2.csv", index_label='PassengerId')
