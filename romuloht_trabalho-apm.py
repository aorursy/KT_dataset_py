import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import os

print(os.listdir("../input"))
train = pd.read_csv('../input/emnist-balanced-train.csv', header=None)
test = pd.read_csv('../input/emnist-balanced-test.csv', header=None)
train.head()
train_data = train.values[:, 1:]

train_labels = train.values[:, 0]

test_data = test.values[:, 1:]

test_labels = test.values[:, 0]


img_flip = np.transpose(train_data[8].reshape(28, 28), axes=[1,0])

plt.imshow(img_flip, cmap='Greys_r')

plt.show()
from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn import metrics
model = {}

y_pred = {}
model['perceptron'] = Perceptron()

model['sgd'] = SGDClassifier()

model['knn'] = KNeighborsClassifier()

model['nbayes'] = GaussianNB()

model['tree'] = DecisionTreeClassifier()

model['forest'] = RandomForestClassifier()

model['boosting'] = GradientBoostingClassifier()
for idx in ['perceptron', 'sgd', 'knn', 'nbayes', 'tree', 'forest', 'boosting']:

    model[idx].fit(train_data, train_labels)

    y_pred[idx] = model[idx].predict(test_data)

    print(idx, metrics.accuracy_score(test_labels, y_pred[idx]))