import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
train_path = os.path.join("../input", "train.csv")
digits = pd.read_csv(train_path)
digits.head(5)
X = digits.drop('label', axis=1)
y = digits['label'].copy()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
mlpc = MLPClassifier(solver='lbfgs', random_state=0, hidden_layer_sizes=[800, ], max_iter=500, tol=0.0001)
mlpc.fit(X_train_scaled, y_train)
y_nn_train_predict = mlpc.predict(X_train_scaled)
confusion_matrix(y_train, y_nn_train_predict)
X_test_scaled = scaler.transform(X_test)
y_nn_test_predict = mlpc.predict(X_test_scaled)
confusion_matrix(y_test, y_nn_test_predict)
mlpc.score(X_test_scaled, y_test)
