import pandas as pd

import numpy as np

import matplotlib as plt
data_frame = pd.read_csv("../input/fraud-rate/main_data.csv")
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import roc_auc_score
frauded_cards = data_frame[data_frame["PotentialFraud"] == 1]

safe_cards = data_frame[data_frame["PotentialFraud"] == 0]
X = data_frame.drop(["PotentialFraud"], axis=1)

Y = data_frame["PotentialFraud"]



X_values = X.values

Y_values = Y.values
X = data_frame.drop(["PotentialFraud"], axis=1)

Y = data_frame["PotentialFraud"]



X_values = X.values

Y_values = Y.values



X_train, X_test, Y_train, Y_test = train_test_split(X_values, 

                                                    Y_values, 

                                                    test_size = 0.25)
safe_cards_undersampling = safe_cards.sample(len(frauded_cards))



undersampled_data = pd.concat([safe_cards_undersampling, frauded_cards], axis = 0)



undersampled_data.PotentialFraud.value_counts().plot(kind='bar', title='Count (PotentialFraud)', color = "red");
frauded_cards_oversampled = frauded_cards.sample(len(safe_cards), replace = True)



oversampled_data = pd.concat([frauded_cards_oversampled, safe_cards], axis = 0)

oversampled_data

oversampled_data.PotentialFraud.value_counts().plot(kind='bar', title='Count (PotentialFraud)')
X_values_u = undersampled_data.drop("PotentialFraud", axis = 1).values

Y_values_u = undersampled_data.PotentialFraud.values



X_train_u, X_test_u, Y_train_u, Y_test_u = train_test_split(X_values_u, Y_values_u, test_size = .25)



RANDOM_FOREST_UNDERSAMPLING = RandomForestClassifier()

RANDOM_FOREST_UNDERSAMPLING.fit(X_train_u, Y_train_u)

Y_u_predict = RANDOM_FOREST_UNDERSAMPLING.predict(X_test_u)

roc_auc_score(Y_u_predict, Y_test_u)
X_values_o = oversampled_data.drop("PotentialFraud", axis = 1).values

Y_values_o = oversampled_data.PotentialFraud.values



X_train_o, X_test_o, Y_train_o, Y_test_o = train_test_split(X_values_o, Y_values_o, test_size = .25)



RANDOM_FOREST_OVERSAMPLING = RandomForestClassifier()

RANDOM_FOREST_OVERSAMPLING.fit(X_train_o, Y_train_o)

Y_o_predict = RANDOM_FOREST_OVERSAMPLING.predict(X_test_o)

roc_auc_score(Y_o_predict, Y_test_o)