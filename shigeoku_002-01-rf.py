import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import tree

# pickle ファイルから読み込み
train_pkl = pd.read_pickle('./pd_train.pk2')
train_pkl.shape
# Age, Fare は削除
# ダミー変数の先頭は削除
train_pkl.drop(['Age', 'Fare', 'Sex_female', 'Embarked_C', 'Pclass_1'], inplace=True, axis=1)
train_pkl.describe()
# 訓練とテストデータに分割
train, test = train_test_split(train_pkl, test_size=0.2, random_state=42)
# ターゲットと特徴量の分割
train_X = train.iloc[:, 1:].values
train_y = train.Survived.values
RF = RandomForestClassifier(random_state=42)
RF = RF.fit(train_X, train_y)
RF.feature_importances_
sorted(
    zip(map(lambda x: round(x, 3), RF.feature_importances_), train.iloc[:, 1:].columns),
    reverse=True)
# 訓練済みの決定木を視覚化
# dot_data = tree.export_graphviz(RF, out_file=None,
#                                feature_names=train.iloc[:, 1:].columns,
#                                class_names=train.Survived.name,
#                                rounded=True,
#                                filled=True,
#                                special_characters=True)
# ターゲットと特徴量の分割
test_x = test.iloc[:, 1:].values
test_y = test.Survived.values
test_x.shape, test_y.shape
pred_y = RF.predict(test_x)
confusion_matrix(test_y, pred_y)
accuracy_score(test_y, pred_y)
# 検証データ読み込み
valid = pd.read_pickle('./pd_test.pk2')
valid.shape
# ID の保存
valid_pass = valid.PassengerId.values
valid_X = valid.iloc[:, 1:]
valid_X.describe()
valid_X.drop(['Age', 'Fare', 'Sex_female', 'Embarked_C', 'Pclass_1'], inplace=True, axis=1)
valid_X.shape, train_X.shape
pred_valid_y = RF.predict(valid_X)
pred_valid_y.shape
type(valid_pass), type(pred_valid_y)
result_df = pd.DataFrame(pred_valid_y, valid_pass, columns=['Survived'])
result_df.to_csv("./RF_1.csv", index_label='PassengerId')
