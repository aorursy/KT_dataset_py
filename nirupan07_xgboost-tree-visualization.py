# plot decision tree

from numpy import loadtxt

from xgboost import XGBClassifier

from xgboost import plot_tree

import matplotlib.pyplot as plt

# load data

dataset = loadtxt('../input/pima_indians.csv', delimiter=",")

# split data into X and y

X = dataset[:,0:8]

y = dataset[:,7]

# fit model no training data

model = XGBClassifier()

model.fit(X, y)
# plot single tree

plot_tree(model, num_trees=1)

plt.show()