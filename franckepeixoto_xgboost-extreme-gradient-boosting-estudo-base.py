import sklearn

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.metrics import precision_score

from sklearn.datasets import load_iris

import xgboost

random_state = 69

dataset = load_iris()

x = dataset.data

y = dataset.target 

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.35, random_state=random_state)
dmatrix_train = xgboost.DMatrix(x_train, label=y_train)

dmatrix_test = xgboost.DMatrix(x_test, label=y_test)
param = { 'max_depth': 3, 'eta': 0.8, 'silent': 1, 'objective': 'multi:softprob', 'num_class': 3} 

num_round = 20
xgb_train = xgboost.train(param,dmatrix_train,num_round)

predictions = xgb_train.predict(dmatrix_test)

best_predictions = np.asarray( [ np.argmax(linha) for linha in predictions])

best_predictions
precision_score(y_test, best_predictions, average='macro')