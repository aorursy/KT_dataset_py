import numpy as np
import pandas as pd

from sklearn import svm
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV

# pickle ファイルから読み込み
train_pkl = pd.read_pickle('./pd_train.pk2')
train_pkl.shape
# Age, Fare は削除
# ダミー変数の先頭は削除
train_pkl.drop(['Age', 'Fare', 'Sex_female', 'Embarked_C', 'Pclass_1'], inplace=True, axis=1)
# 学習が終わらないので、特徴量を上位５つにする
train_pkl.drop(['Parch', 'Embarked_S', 'Embarked_Q'], inplace=True, axis=1)
train_pkl.info()
train_pkl.describe()[['Age_bin', 'Fare_bin']]
train_pkl.Age_bin.max(), train_pkl.Age_bin.min()
train_pkl.Fare_bin.max(), train_pkl.Fare_bin.min()
train_pkl['Age_bin'] = train_pkl['Age_bin'].astype('float16')
train_pkl['Fare_bin'] = train_pkl['Fare_bin'].astype('float16')
train_pkl['SibSp'] = train_pkl['SibSp'].astype('int8')
train_pkl['Sex_male'] = train_pkl['Sex_male'].astype('bool')
train_pkl['Pclass_2'] = train_pkl['Pclass_2'].astype('bool')
train_pkl['Pclass_3'] = train_pkl['Pclass_3'].astype('bool')
train_pkl.info()
# 訓練とテストデータに分割
train, test = train_test_split(train_pkl, test_size=0.2, random_state=42)
# ターゲットと特徴量の分割
train_X = train.iloc[:, 1:].values
train_y = train.Survived.values
grid_param = [
    {'C': [0.001, 0.01, 0.1, 1, 5, 10, 50, 100, 1000],
    'kernel': ['linear', 'rbf'],
    'gamma': [.0001, .001, .01, .1, .5, 1, 10, 30],
#     'shrinking': [True, False],
    'random_state': [42]},
    {'C': [0.001, 0.01, 0.1, 1, 10],
    'gamma': [.0001, .001, .01,.1,.2,.5,1,10],
    'random_state': [42],
    'kernel':['poly']}
]
gs = GridSearchCV(estimator=svm.SVC(grid_param), param_grid=grid_param, scoring='explained_variance', cv=3, return_train_score=False)
gs.fit(train_X, train_y)
gs.best_score_
gs.best_params_
# ターゲットと特徴量の分割
train_X = train.iloc[:, 1:].values
train_y = train.Survived.values
SVC = svm.SVC(**gs.best_params_)
SVC = SVC.fit(train_X, train_y)
# ターゲットと特徴量の分割
test_x = test.iloc[:, 1:].values
test_y = test.Survived.values
test_x.shape, test_y.shape
pred_y = SVC.predict(test_x)
confusion_matrix(test_y, pred_y)
accuracy_score(test_y, pred_y)
# 検証データ読み込み
valid = pd.read_pickle('./pd_test.pk2')
valid.shape
# ID の保存
valid_pass = valid.PassengerId.values
valid_X = valid.iloc[:, 1:]
valid_X.drop(['Age', 'Fare', 'Sex_female', 'Embarked_C', 'Pclass_1'], inplace=True, axis=1)
# 学習が終わらないので、特徴量を上位５つにする
valid_X.drop(['Parch', 'Embarked_S', 'Embarked_Q'], inplace=True, axis=1)
valid_X['Age_bin'] = valid_X['Age_bin'].astype('float16')
valid_X['Fare_bin'] = valid_X['Fare_bin'].astype('float16')
valid_X['SibSp'] = valid_X['SibSp'].astype('int8')
valid_X['Sex_male'] = valid_X['Sex_male'].astype('bool')
valid_X['Pclass_2'] = valid_X['Pclass_2'].astype('bool')
valid_X['Pclass_3'] = valid_X['Pclass_3'].astype('bool')
valid_X.shape, train_X.shape
pred_valid_y = SVC.predict(valid_X)
pred_valid_y.shape
type(valid_pass), type(pred_valid_y)
result_df = pd.DataFrame(pred_valid_y, valid_pass, columns=['Survived'])
result_df.to_csv("./SVC_2.csv", index_label='PassengerId')
