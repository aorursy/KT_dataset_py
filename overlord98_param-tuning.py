import pandas as pd

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score
data_frame = pd.read_csv("../input/fraud-rate/main_data.csv")
X = data_frame.drop(["PotentialFraud"], axis=1)

Y = data_frame["PotentialFraud"]



X_values = X.values

Y_values = Y.values
X_train, X_test, Y_train, Y_test = train_test_split(X_values, 

                                                    Y_values, 

                                                    test_size = 0.25)
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



from sklearn.model_selection import RandomizedSearchCV



rd = RandomForestClassifier()

rdr = RandomizedSearchCV(estimator = rd, param_distributions = grid_param, n_iter = 100, cv = 5, verbose = 2, n_jobs = -1)



rdr.fit(X_train, Y_train)

print(rdr.best_params_)
n_estimators = [80, 85, 90, 95]

min_samples_split = [3, 4, 5, 6, 7, 8 , 9]



grid_param = {

    "n_estimators" : n_estimators,

    "min_samples_split" : min_samples_split

}



rd = RandomForestClassifier()

rdr = RandomizedSearchCV(estimator = rd, param_distributions = grid_param, n_iter = 100, cv = 5, verbose = 2, n_jobs = -1)



rdr.fit(X_train, Y_train)

print(rdr.best_params_)
n_estimators = [90, 92, 94, 96, 98, 100, 102]



grid_param = {

    "n_estimators" : n_estimators,

}



rd = RandomForestClassifier()

rdr = RandomizedSearchCV(estimator = rd, param_distributions = grid_param, n_iter = 100, cv = 5, verbose = 2, n_jobs = -1)



rdr.fit(X_train, Y_train)

print(rdr.best_params_)
n_estimators = [90, 91, 92, 93, 94, 95]



grid_param = {

    "n_estimators" : n_estimators,

}



rd = RandomForestClassifier()

rdr = RandomizedSearchCV(estimator = rd, param_distributions = grid_param, n_iter = 100, cv = 5, verbose = 2, n_jobs = -1)



rdr.fit(X_train, Y_train)

print(rdr.best_params_)