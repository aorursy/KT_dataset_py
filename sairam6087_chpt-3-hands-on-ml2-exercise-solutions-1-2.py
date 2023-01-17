import pandas as pd

import numpy as np

import sklearn

import os

%matplotlib inline

import matplotlib as mpl

import matplotlib.pyplot as plt



np.random.seed(42)



test = pd.read_csv("../input/mnist-in-csv/mnist_test.csv")

train = pd.read_csv("../input/mnist-in-csv/mnist_train.csv")
X_train, y_train = train.drop(labels=["label"],axis=1),  train["label"]

X_test, y_test = test.drop(labels=["label"],axis=1),  test["label"]
X_train = X_train.values.reshape(-1, 784)

X_test = X_test.values.reshape(-1, 784)
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import GridSearchCV

param_grid = [{'weights': ["distance"], 'n_neighbors': [3, 4, 5]}]

knn_clf = KNeighborsClassifier()

grid_search = GridSearchCV(knn_clf, param_grid, cv=5, verbose=3, n_jobs=-1)

grid_search.fit(X_train, y_train)
# What are the best parameters?

grid_search.best_params_
# What's the best score?

grid_search.best_score_
# Did we achieve the target?

from sklearn.metrics import accuracy_score



y_pred = grid_search.predict(X_test)

accuracy_score(y_test, y_pred)
from scipy.ndimage.interpolation import shift



def shift_image(image, dx, dy):

    image = image.reshape((28, 28))

    shifted_image = shift(image, [dy, dx], cval=0, mode="constant")

    return shifted_image.reshape([-1])
image = X_train[1000]

shifted_image_down = shift_image(image, 0, 5)

shifted_image_left = shift_image(image, -5, 0)



plt.figure(figsize=(12,3))

plt.subplot(131)

plt.title("Original", fontsize=14)

plt.imshow(image.reshape(28, 28), interpolation="nearest", cmap="Greys")

plt.subplot(132)

plt.title("Shifted down", fontsize=14)

plt.imshow(shifted_image_down.reshape(28, 28), interpolation="nearest", cmap="Greys")

plt.subplot(133)

plt.title("Shifted left", fontsize=14)

plt.imshow(shifted_image_left.reshape(28, 28), interpolation="nearest", cmap="Greys")

plt.show()
X_train_augmented = [image for image in X_train]

y_train_augmented = [label for label in y_train]



for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):

    for image, label in zip(X_train, y_train):

        X_train_augmented.append(shift_image(image, dx, dy))

        y_train_augmented.append(label)



X_train_augmented = np.array(X_train_augmented)

y_train_augmented = np.array(y_train_augmented)
shuffle_idx = np.random.permutation(len(X_train_augmented))

X_train_augmented = X_train_augmented[shuffle_idx]

y_train_augmented = y_train_augmented[shuffle_idx]
knn_clf = KNeighborsClassifier(**grid_search.best_params_)
knn_clf.fit(X_train_augmented, y_train_augmented)
y_pred = knn_clf.predict(X_test)

accuracy_score(y_test, y_pred)