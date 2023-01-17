import sys

import pandas as pd

import numpy as np

# import pandas

# from pandas.plotting import scatter_matrix

# import matplotlib.pyplot as plt

from sklearn import model_selection

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier



print(sys.version)

#Read data

df_train = pd.read_csv("../input/learn-together/train.csv")

df_test = pd.read_csv("../input/learn-together/test.csv")

full_features = list(df_train.columns)

n_ff = len(full_features)

a_feature = []

a_feature.append(full_features[-1])

df_X = df_train[full_features[0: n_ff - 1]]

df_Y = df_train[a_feature]

print(df_X.columns)

print(df_Y.columns)

X = df_X.values

Y = df_Y.values



validation_size = 0.3

seed = 7

X_train, X_validation, Y_train, Y_validation = train_test_split (

                         X, Y, test_size = validation_size, random_state = seed )



model = DecisionTreeClassifier(random_state = 1)

model.fit(X_train, Y_train)

predictions = model.predict(X_validation)

print("Accuracy:" , accuracy_score(Y_validation, predictions))

print("\nConfusion Matrix:\n", confusion_matrix(Y_validation, predictions))

print("\nDetails:\n", classification_report(Y_validation, predictions))

print("-----------------------------------------------------------------------\n")








