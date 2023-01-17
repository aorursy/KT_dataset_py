import numpy as np

import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import train_test_split
raw_data = pd.read_csv("../input/adult.csv")

dummies = pd.get_dummies(raw_data)

del dummies["income_<=50K"]

XY = dummies.values

X = XY[:,:-1]

Y = XY[:,-1]

print(X.shape, Y.shape)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

gbc = GradientBoostingClassifier().fit(X_train, Y_train)
print("GBC %s" % gbc.score(X_test, Y_test))