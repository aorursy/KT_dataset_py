!pip install ultimate==2.36.2
# -*- coding: utf-8 -*-
from __future__ import print_function
from ultimate.mlp import MLP
from ultimate.plotting import plot_importance

import numpy as np
import sys, random
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

iris = load_iris()

train_in = np.array(iris['data'], dtype=np.float32)
train_out = np.array(iris['target'], dtype=np.float32)
random_state = 42 

X_train, X_test, y_train, y_test = train_test_split(
    train_in, train_out, test_size=0.2, random_state=random_state)

param = {
    'loss_type': 'softmax',
    'layer_size': [4, 8, 8, 8, 3],
    'activation': 'am2',
    'output_range': [0, 1],
    'output_shrink': 0.001,
    'importance_out': True,
    'rate_init': 0.02, 
    'rate_decay': 0.9, 
    'epoch_train': 2000, 
    'epoch_decay': 40,
    'verbose': 0,
}

mlp = MLP(param).fit(X_train, y_train)

plot_importance(mlp.feature_importances_, feature_names=iris.feature_names)

pred = mlp.predict(X_test)

print('category:', pred)
print('probability:', mlp.predict_proba(X_test))

score = accuracy_score(y_test, pred)

print("accuracy score: %.2f%%" % (score*100))
