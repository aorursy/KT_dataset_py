import numpy as np
import pandas as pd

import xgboost
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

# pickle ファイルから読み込み
train_pkl = pd.read_pickle('./pd_train.pk2')
train_pkl.shape
# Age, Fare は削除
# ダミー変数の先頭は削除
train_pkl.drop(['Age', 'Fare', 'Sex_female', 'Embarked_C', 'Pclass_1'], inplace=True, axis=1)
# 訓練とテストデータに分割
train, test = train_test_split(train_pkl, test_size=0.2, random_state=42)
# ターゲットと特徴量の分割
# train_X = train.iloc[:, 1:].values
# train_y = train.Survived.values
# ターゲットと特徴量の分割
train_X = train.iloc[:, 1:]
train_y = train.Survived
XGB = XGBClassifier()
XGB = XGB.fit(train_X, train_y)
XGB.feature_importances_
sorted(
    zip(map(lambda x: round(x, 3), XGB.feature_importances_), train.iloc[:, 1:].columns),
    reverse=True)
# ターゲットと特徴量の分割
# test_x = test.iloc[:, 1:].values
# test_y = test.Survived.values
# ターゲットと特徴量の分割
test_x = test.iloc[:, 1:]
test_y = test.Survived
test_x.shape, test_y.shape
pred_y = XGB.predict(test_x)
confusion_matrix(test_y, pred_y)
accuracy_score(test_y, pred_y)
# 検証データ読み込み
valid = pd.read_pickle('./pd_test.pk2')
valid.shape
# ID の保存
valid_pass = valid.PassengerId.values
valid_X = valid.iloc[:, 1:]
valid_X.drop(['Age', 'Fare', 'Sex_female', 'Embarked_C', 'Pclass_1'], inplace=True, axis=1)
valid_X.shape, train_X.shape
# valid_X_2 = valid_X.rename(columns={'SibSp':'f0'})
# valid_X_2.rename(columns={'Parch':'f1', 'Age_bin':'f2', 'Fare_bin':'f3', 'Sex_male':'f4', 'Embarked_Q':'f5', 'Embarked_S':'f6', 'Pclass_2':'f7', 'Pclass_3':'f8'}, inplace=True)
pred_valid_y = XGB.predict(valid_X)
pred_valid_y.shape
type(valid_pass), type(pred_valid_y)
result_df = pd.DataFrame(pred_valid_y, valid_pass, columns=['Survived'])
result_df.to_csv("./XGB_1.csv", index_label='PassengerId')
