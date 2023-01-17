import pandas as pd

import numpy as np
from sklearn.datasets import fetch_openml

import joblib



# mnist = fetch_openml('mnist_784')

mnist = joblib.load('/kaggle/input/written-numbers/mnist.pkl')
print(mnist.DESCR)
X, y = mnist.data, mnist.target

X.shape
y.shape
%matplotlib inline

import matplotlib

import matplotlib.pyplot as plt



def show_digit(pixels, plt=plt):

    pixels = pixels.reshape(28, 28)

    plt.imshow(pixels, cmap=matplotlib.cm.binary, interpolation='nearest')

    plt.axis('off')

    plt.show()
show_digit(X[0])
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
shuffle_idx = np.random.permutation(60000)

X_train, y_train = X_train[shuffle_idx], y_train[shuffle_idx]
y_train_5 = y_train == '5'

y_test_5 = y_test == '5'
from sklearn.linear_model import SGDClassifier



sgd_clf = SGDClassifier(random_state=42)

sgd_clf.fit(X_train, y_train_5)
sgd_clf.predict([X_train[0]])
from sklearn.model_selection import cross_val_score



cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring='accuracy')
len(y_train_5[y_train_5]) / len(y_train_5)
from sklearn.model_selection import cross_val_predict



y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)

y_train_pred
from sklearn.metrics import confusion_matrix



confusion_matrix(y_train_5, y_train_pred)
from sklearn.metrics import precision_score, recall_score



precision_score(y_train_5, y_train_pred)
recall_score(y_train_5, y_train_pred)
from sklearn.metrics import f1_score



f1_score(y_train_5, y_train_pred)
scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method='decision_function')
from sklearn.metrics import precision_recall_curve



precisions, recalls, thresholds = precision_recall_curve(y_train_5, scores)
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds, plt=plt):

    plt.plot(thresholds, precisions[:-1], 'b--', label='Precision')

    plt.plot(thresholds, recalls[:-1], 'g-', label='Recall')

    plt.xlabel('Threshold')

    plt.legend(loc='upper left')

    plt.ylim([0, 1])

    

def plot_precision_vs_recall(precisions, recalls, plt=plt):

    plt.plot(recalls, precisions)

    plt.xlabel('Recall')

    plt.ylabel('Precision')

    plt.xlim([0, 1])

    plt.ylim([0, 1])
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)

plt.show()
plot_precision_vs_recall(precisions, recalls)

plt.show()
from sklearn.metrics import roc_curve



fpr, tpr, thresholds = roc_curve(y_train_5, scores)
def plot_roc_curve(fpr, tpr, label=None, plt=plt):

    plt.plot(fpr, tpr, linewidth=2, label=label)

    plt.plot([0, 1], [0, 1], 'k--')

    plt.axis([0, 1, 0, 1])

    plt.xlabel('False Positive Rate (FPR)')

    plt.ylabel('True Positive Rate (TPR)')
plot_roc_curve(fpr, tpr)

plt.show()
from sklearn.metrics import roc_auc_score



roc_auc_score(y_train_5, scores)
sgd_clf.fit(X_train, y_train)
sgd_clf.predict([X[0]])
sgd_clf.decision_function([X[0]])
sgd_clf.classes_
from sklearn.ensemble import RandomForestClassifier



for_clf = RandomForestClassifier()

for_clf.fit(X_train, y_train)
# for digit in X[:3]:

#     show_digit(digit)
for_clf.predict(X[:3])
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
y_train_pred = cross_val_predict(for_clf, X_train_scaled, y_train, cv=3)

conf_mx = confusion_matrix(y_train, y_train_pred)

conf_mx
plt.matshow(conf_mx, cmap=plt.cm.gray)

plt.show()
row_sums = conf_mx.sum(axis=1, keepdims=True)

norm_conf_mx = conf_mx / row_sums

np.fill_diagonal(norm_conf_mx, 0)

plt.matshow(norm_conf_mx, cmap=plt.cm.gray)

plt.show()
from sklearn.neighbors import KNeighborsClassifier



y_train_large = y_train.astype(np.float) >= 7

y_train_odd = y_train.astype(np.float) % 2 == 1

y_multilabel = np.c_[y_train_large, y_train_odd]



knn_clf = KNeighborsClassifier()

knn_clf.fit(X_train, y_multilabel)
knn_clf.predict([X_train[-1]])
show_digit(X_train[-1])
# y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_multilabel, cv=3)
# f1_score(y_train, y_train_knn_pred, average='macro')
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline



clf = Pipeline([

    ('std_scaler', StandardScaler()),

    ('knn_clf', KNeighborsClassifier()),

])
clf.fit(X_train, y_train)
scores = cross_val_score(

    clf,

    X_train,

    y_train,

    cv=3,

    scoring='accuracy',

    n_jobs=-1)

scores