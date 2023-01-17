import pandas as pd

from sklearn.model_selection import cross_val_score, cross_validate

from sklearn import tree

from sklearn.metrics import accuracy_score

import numpy as np

import matplotlib.pyplot as plt

from sklearn.externals.six import StringIO

import pydot
trainData = pd.read_csv('../input/train.csv')
trainData.head()
trainPriceRange = trainData["price_range"]

trainData = trainData.drop("price_range", axis=1)
clf = tree.DecisionTreeClassifier(random_state=0)

cross_return = cross_validate(clf, trainData, trainPriceRange, cv=10, return_estimator=True)
cross_return["test_score"]
cross_return["test_score"].mean()
min_test = min(cross_return["test_score"])



max_test = max(cross_return["test_score"])

max_index = [i for i, j in enumerate(cross_return["test_score"]) if j == max_test]
best_tree = cross_return["estimator"][max_index[0]]
best_tree.feature_importances_
height = best_tree.feature_importances_

bars = trainData.columns

y_pos = np.arange(len(bars))

 

plt.bar(y_pos, height)

plt.xticks(y_pos, bars, rotation = 90)

plt.show()