import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import tree

# pickle ファイルから読み込み
train_pkl = pd.read_pickle('./pd_train.pk2')
train_pkl.shape
# Age, Fare は削除
# ダミー変数の先頭は削除
train_pkl.drop(['Age', 'Fare', 'Sex_female', 'Embarked_C', 'Pclass_1'], inplace=True, axis=1)
# 訓練とテストデータに分割
train, test = train_test_split(train_pkl, test_size=0.2, random_state=42)
# ターゲットと特徴量の分割
train_X = train.iloc[:, 1:].values
train_y = train.Survived.values
grid_param = {
    'n_estimators': [10, 30, 50, 70, 90, 100, 110, 120],
    'criterion': ['gini', 'entropy'],
    'max_depth': [i for i in range(1, 11)],
    'min_samples_split': [i for i in range(2, 11)],
    'min_samples_leaf': [i for i in range(1, 11)],
    'random_state': [42]
}

gs = GridSearchCV(estimator=RandomForestClassifier(**grid_param), param_grid=grid_param, scoring='accuracy', cv=5, return_train_score=False)
gs.fit(train_X, train_y)
gs.best_score_
gs.best_params_
RF = RandomForestClassifier(**gs.best_params_)
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
result_df.to_csv("./RF_3.csv", index_label='PassengerId')
