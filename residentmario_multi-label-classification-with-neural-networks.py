import numpy as np
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=10000, n_features=5, n_informative=5,
                           n_redundant=0, n_repeated=0, n_classes=3,
                           n_clusters_per_class=1,
                           weights=[0.33, 0.33, 0.33],
                           class_sep=2, random_state=0)

import matplotlib.pyplot as plt
colors = ['#ef8a62' if v == 0 else '#f7f7f7' if v == 1 else '#67a9cf' for v in y]
kwarg_params = {'linewidth': 1, 'edgecolor': 'black'}
plt.scatter(X[:, 0], X[:, 1], c=colors, **kwarg_params)
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.metrics import binary_accuracy
from keras.utils import to_categorical

X, y = make_classification(n_samples=10000, n_features=5, n_informative=5,
                           n_redundant=0, n_repeated=0, n_classes=3,
                           n_clusters_per_class=1,
                           weights=[0.33, 0.33, 0.33],
                           class_sep=2, random_state=0)
y = to_categorical(y)
y = np.vstack((y[:, 0], y[:, :2].sum(axis=1))).T

clf = Sequential()
clf.add(Dense(5, activation='relu', input_dim=5))
clf.add(Dense(2, activation='sigmoid'))
clf.compile(loss='binary_crossentropy', optimizer=SGD(), metrics=[binary_accuracy])
clf.fit(X, y, epochs=20, batch_size=100, verbose=0)
clf.predict(X)
y
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.metrics import binary_accuracy
from keras.utils import to_categorical

X, y = make_classification(n_samples=10000, n_features=5, n_informative=5,
                           n_redundant=0, n_repeated=0, n_classes=3,
                           n_clusters_per_class=1,
                           weights=[0.33, 0.33, 0.33],
                           class_sep=2, random_state=0)
y = to_categorical(y)
y = np.vstack((y[:, 0], y[:, :2].sum(axis=1))).T

clf = Sequential()
clf.add(Dense(5, activation='relu', input_dim=5))
clf.add(Dense(2, activation='softmax'))
clf.compile(loss='categorical_crossentropy', optimizer=SGD(), metrics=[binary_accuracy])
clf.fit(X, y, epochs=20, batch_size=100, verbose=0)
clf.predict(X)
y