import numpy as np
import pandas as pd

from sklearn.neural_network import MLPClassifier
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
# 訓練とテストデータに分割
train, test = train_test_split(train_pkl, test_size=0.2, random_state=42)
# ターゲットと特徴量の分割
train_X = train.iloc[:, 1:].values
train_y = train.Survived.values
grid_param = [
    {
    'solver': ['sgd'],    
    'activation': ['logistic', 'relu'],
    'hidden_layer_sizes':[3, 5, 6, 7, 10, 15],
    'learning_rate_init': [0.001, 0.01, 0.1, 1],
    'alpha': [0.1, 0.01, 0.001, 0.0001, 0.00001],
#     'max_iter': [],
    'random_state': [42]
    },
    {
    'solver': ['adam'],    
    'activation': ['logistic', 'relu'],
    'hidden_layer_sizes':[3, 5, 6, 7, 10, 15],
    'learning_rate_init': [0.001, 0.01, 0.1, 1],
    'alpha': [0.1, 0.01, 0.001, 0.0001, 0.00001],
#     'max_iter': [],
    'random_state': [42]
    }
]

gs = GridSearchCV(estimator=MLPClassifier(), param_grid=grid_param, scoring='accuracy', cv=5, return_train_score=False)
gs.fit(train_X, train_y)
gs.best_score_
gs.best_params_
gs.cv_results_['mean_test_score']
NN = MLPClassifier(**gs.best_params_)
NN = NN.fit(train_X, train_y)
# ターゲットと特徴量の分割
test_x = test.iloc[:, 1:].values
test_y = test.Survived.values
test_x.shape, test_y.shape
pred_y = NN.predict(test_x)
confusion_matrix(test_y, pred_y)
accuracy_score(test_y, pred_y)
# 検証データ読み込み
valid = pd.read_pickle('./pd_test.pk2')
valid.shape
# ID の保存
valid_pass = valid.PassengerId.values
valid_X = valid.iloc[:, 1:]
valid_X.drop(['Age', 'Fare', 'Sex_female', 'Embarked_C', 'Pclass_1'], inplace=True, axis=1)
valid_X.drop(['Parch', 'Embarked_S', 'Embarked_Q'], inplace=True, axis=1)
valid_X.shape, train_X.shape
pred_valid_y = NN.predict(valid_X)
pred_valid_y.shape
type(valid_pass), type(pred_valid_y)
result_df = pd.DataFrame(pred_valid_y, valid_pass, columns=['Survived'])
result_df.to_csv("./NN_3.csv", index_label='PassengerId')
