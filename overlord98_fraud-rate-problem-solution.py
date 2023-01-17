import numpy as np

import pandas as pd

import matplotlib as plt
data_frame = pd.read_csv("../input/fraud-rate/main_data.csv")
data_frame.head()
data_frame.shape
data_frame.columns
X = data_frame.drop(["PotentialFraud"], axis=1)

Y = data_frame["PotentialFraud"]



X_values = X.values

Y_values = Y.values
from sklearn.model_selection import train_test_split



X_train, X_test, Y_train, Y_test = train_test_split(X_values, 

                                                    Y_values, 

                                                    test_size = 0.25,

                                                    random_state = 1)
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score



RANDOM_FOREST = RandomForestClassifier()
RANDOM_FOREST.fit(X_train, Y_train)
Y_predict = RANDOM_FOREST.predict(X_test)
frauded_cards = data_frame[data_frame["PotentialFraud"] == 1]

safe_cards = data_frame[data_frame["PotentialFraud"] == 0]



part_of_frauded_cards = 100 * len(frauded_cards)/len(data_frame)

min_accurancy = 100 - part_of_frauded_cards

round(min_accurancy, 3)
from sklearn.metrics import roc_auc_score



r_accuracy = roc_auc_score(Y_predict, Y_test)

round(r_accuracy, 3)
from sklearn.model_selection import RandomizedSearchCV
n_estiamtors = [5, 10, 30, 50, 70, 90, 110, 130, 150, 170]

max_features = ["auto", "sqrt", "log2"]

max_depth = [5, 7, 8, 9, 10, 11, 12]

min_samples_split = [2, 5, 10, 15, 20]

min_samples_leaf = [1, 2, 5, 10, 15]
grid_param = {

    "n_estimators" : n_estiamtors,

    "max_features" : max_features,

    "max_depth" : max_depth,

    "min_samples_split" : min_samples_split,

    "min_samples_leaf" : min_samples_leaf

}
rd = RandomForestClassifier()

rdr = RandomizedSearchCV(estimator = rd, param_distributions = grid_param, n_iter = 100, cv = 5, verbose = 2, n_jobs = -1)

rdr.fit(X_train, Y_train)
rdr.best_params_
n_estiamtors = [145, 150, 155, 160]

max_depth = [11, 12, 13]

min_samples_split = [1, 2, 3]
grid_param = {

    "n_estimators" : n_estiamtors,

    "max_depth" : max_depth,

    "min_samples_split" : min_samples_split

}
rd = RandomForestClassifier()

rdr = RandomizedSearchCV(estimator = rd, param_distributions = grid_param, n_iter = 100, cv = 5, verbose = 2, n_jobs = -1)

rdr.fit(X_train, Y_train)
rdr.best_params_
n_estiamtors = [152, 154, 155, 156, 157]

min_samples_split = [3, 4, 5]
grid_param = {

    "n_estimators" : n_estiamtors,

    "min_samples_split" : min_samples_split

}
rd = RandomForestClassifier(max_depth = 12, min_samples_leaf = 1, max_features = 'auto')

rdr = RandomizedSearchCV(estimator = rd, param_distributions = grid_param, n_iter = 100, cv = 5, verbose = 2, n_jobs = -1)

rdr.fit(X_train, Y_train)
rdr.best_params_
n_estiamtors = [155, 156, 157, 158]
grid_param = {

    "n_estimators" : n_estiamtors

}
rd = RandomForestClassifier(max_depth = 12, min_samples_leaf = 1, max_features = 'auto', min_samples_split = 4)

rdr = RandomizedSearchCV(estimator = rd, param_distributions = grid_param, n_iter = 100, cv = 5, verbose = 2, n_jobs = -1)

rdr.fit(X_train, Y_train)
rdr.best_params_
#Побудуємо класифікатор із знайденими параметрами
RANDOM_FOREST_WITH_BEST_PARAM = RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,

                       criterion='entropy', max_depth=10, max_features='log2',

                       max_leaf_nodes=60, max_samples=None,

                       min_impurity_decrease=0.0, min_impurity_split=None,

                       min_samples_leaf=10, min_samples_split=3,

                       min_weight_fraction_leaf=0.0, n_estimators=85,

                       n_jobs=None, oob_score=False, random_state=None,

                       verbose=0, warm_start=False)
RANDOM_FOREST_WITH_BEST_PARAM.fit(X_train, Y_train)
Y_predict = RANDOM_FOREST_WITH_BEST_PARAM.predict(X_test)
round(roc_auc_score(Y_predict, Y_test), 3)
round(max(cross_val_score(RANDOM_FOREST_WITH_BEST_PARAM, X_train, Y_train, cv=10)), 3)
frauded_cards_oversampled = frauded_cards.sample(len(safe_cards), replace = True)



oversampled_data = pd.concat([frauded_cards_oversampled, safe_cards], axis = 0)
X_values_o = oversampled_data.drop("PotentialFraud", axis = 1).values

Y_values_o = oversampled_data.PotentialFraud.values



X_train_o, X_test_o, Y_train_o, Y_test_o = train_test_split(X_values_o, Y_values_o, test_size = .25)
RANDOM_FOREST_OVERSAMPLING = RandomForestClassifier()

RANDOM_FOREST_OVERSAMPLING.fit(X_train_o, Y_train_o)

Y_o_predict = RANDOM_FOREST_OVERSAMPLING.predict(X_test_o)

roc_auc_score(Y_o_predict, Y_test_o)

cross_val_score(RANDOM_FOREST_OVERSAMPLING, X_train_o, Y_train_o, cv=5).mean()
n_estiamtors = [5, 10, 30, 50, 70, 90, 110, 130, 150, 170]

max_features = ["auto", "sqrt", "log2"]

max_depth = [5, 7, 8, 9, 10, 11, 12]

min_samples_split = [2, 5, 10, 15, 20]

min_samples_leaf = [1, 2, 5, 10, 15]
grid_param = {

    "n_estimators" : n_estiamtors,

    "max_features" : max_features,

    "max_depth" : max_depth,

    "min_samples_split" : min_samples_split,

    "min_samples_leaf" : min_samples_leaf

}
rd = RandomForestClassifier()

rdr = RandomizedSearchCV(estimator = rd, param_distributions = grid_param, n_iter = 100, cv = 5, verbose = 2, n_jobs = -1)

rdr.fit(X_train_o, Y_train_o)
rdr.best_params_
n_estiamtors = [120, 125, 130, 135, 140]

max_depth = [11, 12, 13]

min_samples_split = [3, 4, 5, 6, 7]
grid_param = {

    "n_estimators" : n_estiamtors,

    "max_depth" : max_depth,

    "min_samples_split" : min_samples_split

}
rd = RandomForestClassifier(min_samples_leaf = 1, max_features = "sqrt")

rdr = RandomizedSearchCV(estimator = rd, param_distributions = grid_param, n_iter = 100, cv = 5, verbose = 2, n_jobs = -1)

rdr.fit(X_train_o, Y_train_o)
rdr.best_params_
n_estiamtors = [118, 119, 120, 121, 120]
grid_param = {

    "n_estimators" : n_estiamtors

}
rd = RandomForestClassifier(min_samples_leaf = 1, max_features = "sqrt", min_samples_split = 4, max_depth = 12)

rdr = RandomizedSearchCV(estimator = rd, param_distributions = grid_param, n_iter = 100, cv = 5, verbose = 2, n_jobs = -1)

rdr.fit(X_train_o, Y_train_o)
rdr.best_params_
RANDOM_FOREST_OVERSAMPLING_WITH_BEST_PARAMS = RandomForestClassifier(n_estimators = 120,

                                                                    max_depth = 12, 

                                                                    min_samples_leaf = 1,

                                                                    max_features = "sqrt",

                                                                    min_samples_split = 4)
RANDOM_FOREST_OVERSAMPLING_WITH_BEST_PARAMS.fit(X_train_o, Y_train_o)
Y_predict_o = RANDOM_FOREST_OVERSAMPLING_WITH_BEST_PARAMS.predict(X_test_o)
round(roc_auc_score(Y_predict_o, Y_test_o), 2)
cross_val_score(RANDOM_FOREST_OVERSAMPLING_WITH_BEST_PARAMS, X_train_o, Y_train_o, cv=5).mean()