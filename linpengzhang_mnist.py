# load modules

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import math

from sklearn.metrics import accuracy_score

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.svm import SVC

from time import perf_counter

import warnings

warnings.filterwarnings('ignore')



np.random.seed(0)
# caricamento dataset

test  = pd.read_csv('../input/mnist-in-csv/mnist_test.csv')

train = pd.read_csv('../input/mnist-in-csv/mnist_train.csv')

print(train.shape)

print(test.shape)

train.head()
y_train = train['label'][:1000]

y_test = test['label'][:500]

x_train = train.drop('label', axis=1).values[:1000]

x_test = test.drop('label', axis=1).values[:500]
def gridsearch_and_print(clf):

    t0 = perf_counter()

    clf.fit(x_train, y_train)

    print("Grid Search done in %0.3fs" % (perf_counter() - t0))

    print("TRAINING score:", clf.best_score_)

    print("BEST parameters:", clf.best_params_)
p_grid = {

    "svm": [

        {"C": [3 ** i for i in range(-7,-5)], "kernel": ["poly"], "gamma": [3 ** i for i in range(-5, -2)],

         "degree": [i for i in range(1, 4)]}

    ],

    "knn": {

        "n_neighbors": [i for i in range(1, 5)]

    },

    "rforest": {

        'max_depth': [i * 2 for i in range(4, 7)],

        'min_samples_leaf': [i for i in range(2, 6)],

        'min_samples_split': [i for i in range(2, 6)],

        'n_estimators': [3 ** i for i in range(3, 6)]

    },

    "mlp": {

        'activation': ['logistic', 'relu'],

        'hidden_layer_sizes': [(100,), (150,), (100, 50,), (150, 50,)]

    }

}
clf = GridSearchCV(KNeighborsClassifier(), param_grid=p_grid["knn"], cv=5, scoring='accuracy')

gridsearch_and_print(clf)
clf = GridSearchCV(RandomForestClassifier(), param_grid=p_grid["rforest"], cv=5, scoring='accuracy')

gridsearch_and_print(clf)
clf = GridSearchCV(SVC(), param_grid=p_grid["svm"], cv=5, scoring='accuracy')

gridsearch_and_print(clf)
clf = GridSearchCV(MLPClassifier(max_iter=1000), param_grid=p_grid["mlp"], cv=5, scoring='accuracy')

gridsearch_and_print(clf)
y_train_F = train['label']

y_test_F = test['label']

x_train_F = train.drop('label', axis=1).values

x_test_F = test.drop('label', axis=1).values



metrics = dict()
start = perf_counter()

knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(x_train_F, y_train_F)

middle = perf_counter()

training_time = middle - start

pred = knn.predict(x_test_F)

prediction_time = perf_counter() - middle

metrics['k Nearest Neighbors'] = [accuracy_score(y_test_F, pred), training_time, prediction_time]

print(

    f"k Nearest Neighbors accuracy:\t {metrics['k Nearest Neighbors'][0]}\t training time: {metrics['k Nearest Neighbors'][1]}\t prediction time: {metrics['k Nearest Neighbors'][2]}")
start = perf_counter()

rf = RandomForestClassifier(max_depth=10, min_samples_leaf=2, min_samples_split=5, n_estimators=243)

rf.fit(x_train_F, y_train_F)

middle = perf_counter()

training_time = middle - start

pred = rf.predict(x_test_F)

prediction_time = perf_counter() - middle

metrics['Random Forest'] = [accuracy_score(y_test_F, pred), training_time, prediction_time]

print(

    f"Random Forest accuracy:\t {metrics['Random Forest'][0]}\t training time: {metrics['Random Forest'][1]}\t prediction time: {metrics['Random Forest'][2]}")
start = perf_counter()

svm = SVC(kernel='poly', C=3 ** (-7), gamma=3 ** (-4), degree=2)

svm.fit(x_train_F, y_train_F)

middle = perf_counter()

training_time = middle - start

pred = svm.predict(x_test_F)

prediction_time = perf_counter() - middle

metrics['SVM'] = [accuracy_score(y_test_F, pred), training_time, prediction_time]

print(

    f"SVM accuracy:\t {metrics['SVM'][0]}\t training time: {metrics['SVM'][1]}\t prediction time: {metrics['SVM'][2]}")
start = perf_counter()

mlp = MLPClassifier(activation='logistic', hidden_layer_sizes=(100,), max_iter=1000)

mlp.fit(x_train_F, y_train_F)

middle = perf_counter()

training_time = middle - start

pred = mlp.predict(x_test_F)

prediction_time = perf_counter() - middle

metrics['Multilayer Perceptron'] = [accuracy_score(y_test_F, pred), training_time, prediction_time]

print(

    f"Multilayer Perceptron accuracy:\t {metrics['Multilayer Perceptron'][0]}\t training time: {metrics['Multilayer Perceptron'][1]} \t prediction time: {metrics['Multilayer Perceptron'][2]}")
estimators = [knn, rf, svm, mlp]

preds = []

pred = []

start = perf_counter()

for clf in estimators:

    preds.append(clf.predict(x_test_F))

for i in range(0, len(x_test_F)):

    counting = [0 for _ in range(10)]

    for apred in preds:

        counting[apred[i]] += 1

    pred.append(np.argmax(counting))

prediction_time = perf_counter() - start

metrics['Ensemble'] = [accuracy_score(y_test_F, pred),

                       sum([metrics['k Nearest Neighbors'][1], metrics['Random Forest'][1], metrics['SVM'][1], metrics['Multilayer Perceptron'][1]]),

                       prediction_time]

print(

    f"Ensemble accuracy:\t {metrics['Ensemble'][0]}\t training time: {metrics['Ensemble'][1]}\t prediction time: {metrics['Ensemble'][2]}")
tab = pd.DataFrame.from_dict(metrics, orient="index", columns=["accuracy", "training time", "prediction time"])

print(tab.sort_values(by=['accuracy', 'training time', 'prediction time'], ascending=False))