import numpy as np 

import pandas as pd 

from sklearn.model_selection import train_test_split , GridSearchCV

from sklearn.svm import SVC

from sklearn.metrics import classification_report , confusion_matrix

from sklearn.datasets import fetch_lfw_people

import tqdm

import matplotlib.pyplot as plt

import logging

from sklearn.decomposition import PCA
# Display progress logs on stdout

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
lwf_people = fetch_lfw_people(min_faces_per_person=70 , resize=0.4 )
smaples , h ,w =  lwf_people.images.shape
lwf_people.data.shape
X = lwf_people.data

n_features = X.shape[1]
y=lwf_people.target

target_names = lwf_people.target_names

n_class = target_names.shape[0]
print("Total dataset size:")

print("n_samples: %d" % smaples)

print("n_features: %d" % n_features)

print("n_classes: %d" % n_class)
X_train, X_test, y_train, y_test = train_test_split(

    X, y, test_size=0.25, random_state=42)
import time

# Compute a PCA (eigenfaces) on the face dataset (treated as unlabeled

# dataset): unsupervised feature extraction / dimensionality reduction

n_components = 150



print("Extracting the top %d eigenfaces from %d faces"% (n_components, X_train.shape[0]))

t0 = time.time()

pca = PCA(n_components=n_components, svd_solver='randomized',whiten=True).fit(X_train)

print("done in %0.3fs" % (time.time() - t0))
eigenfaces = pca.components_.reshape((n_components , h,w))
print("Projecting the input data on the eigenfaces orthonormal basis")

t0 = time.time()

X_train_pca = pca.transform(X_train)

X_test_pca = pca.transform(X_test)

print("done in %0.3fs" % (time.time() - t0))
# Train a SVM classification model



print("Fitting the classifier to the training set")

t0 = time.time()

param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],

              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }

clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'),

                   param_grid, cv=5, iid=False)

clf = clf.fit(X_train_pca, y_train)

print("done in %0.3fs" % (time.time() - t0))

print("Best estimator found by grid search:")

print(clf.best_estimator_)
# Quantitative evaluation of the model quality on the test set



print("Predicting people's names on the test set")

t0 = time.time()

y_pred = clf.predict(X_test_pca)

print("done in %0.3fs" % (time.time() - t0))



print(classification_report(y_test, y_pred, target_names=target_names))

print(confusion_matrix(y_test, y_pred, labels=range(n_class)))
