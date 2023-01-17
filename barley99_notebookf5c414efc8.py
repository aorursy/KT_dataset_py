import numpy as np

import pandas as pd

from sklearn import neighbors, metrics, cross_validation
data = pd.read_csv('../input/train.csv')

X = data.iloc[:, 1:].copy()

y = data.label.copy()
nnclassifier = neighbors.KNeighborsClassifier(1)
#print(cross_validation.cross_val_score(nnclassifier, X, y, cv=2, scoring='accuracy').mean())
nnclassifier.fit(X, y)