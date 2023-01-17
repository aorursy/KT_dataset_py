import numpy as np

import pandas as pd
from sklearn.metrics import roc_auc_score

from sklearn.model_selection import train_test_split
main_data = pd.read_csv("../input/fraud-rate/main_data.csv")
X = main_data.drop("PotentialFraud", axis = 1)

Y = main_data.PotentialFraud



X = X.values

Y = Y.values



X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = .25)
from sklearn.tree import DecisionTreeClassifier



DECISION_TREE_CLASSIFIER = DecisionTreeClassifier()

DECISION_TREE_CLASSIFIER.fit(X_train, Y_train)

Y_predict = DECISION_TREE_CLASSIFIER.predict(X_test)

round(roc_auc_score(Y_predict, Y_test), 3)
from sklearn.ensemble import IsolationForest



ISOLATION_FOREST = IsolationForest()

ISOLATION_FOREST.fit(X_train, Y_train)

Y_predict = ISOLATION_FOREST.predict(X_test)

round(roc_auc_score(Y_predict, Y_test), 2)
from sklearn.ensemble import GradientBoostingClassifier



GRADIENT_BOOSTING_CLASSIFIER = GradientBoostingClassifier()

GRADIENT_BOOSTING_CLASSIFIER.fit(X_train, Y_train)

Y_predict = GRADIENT_BOOSTING_CLASSIFIER.predict(X_test)

round(roc_auc_score(Y_predict, Y_test), 2)
from sklearn.experimental import enable_hist_gradient_boosting

from sklearn.ensemble import HistGradientBoostingClassifier



HIST_GRADIENT_BOOSTING_CLASSIFIER = HistGradientBoostingClassifier(max_depth = 8,

                                                                      min_samples_leaf = 21,

                                                                      max_leaf_nodes = 33)

HIST_GRADIENT_BOOSTING_CLASSIFIER.fit(X_train, Y_train)

Y_predict = HIST_GRADIENT_BOOSTING_CLASSIFIER.predict(X_test)

round(roc_auc_score(Y_predict, Y_test), 5)