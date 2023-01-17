import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

from random import randint

import string

import csv
dataset = pd.read_csv('../input/data.csv')
dataset
dataset.describe()
dataset = dataset.drop(columns = 'id')
X = dataset.iloc[:,1:18].values
X
Y = dataset.iloc[:,0:1].values
Y
from sklearn.preprocessing import LabelEncoder

labelencoder_Y = LabelEncoder()

Y = labelencoder_Y.fit_transform(Y)
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

X = sc_X.fit_transform(X)

# sc_Y = StandardScaler()

# Y = sc_Y.fit_transform(Y.reshape(-1,1))
X
Y
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 49)
X_train.shape
X_test.shape
Y_train.shape
Y_test.shape
from sklearn.svm import SVC

classifier = SVC(kernel = 'rbf', random_state = 49)

classifier.fit(X_train, Y_train)

Y_pred = classifier.predict(X_test)
Y_test
Y_pred
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(Y_test, Y_pred)
cm
from sklearn.metrics import average_precision_score

score1 = average_precision_score(Y_test,Y_pred)

score1
from sklearn.metrics import balanced_accuracy_score

score2 = balanced_accuracy_score(Y_test,Y_pred)

score2
from sklearn.metrics import accuracy_score

score3 = accuracy_score(Y_test,Y_pred)

score3
from sklearn.metrics import f1_score

score4 = f1_score(Y_test,Y_pred)

score4
final_score = (score1+score2+score3+score4)/4

final_score
from matplotlib.colors import ListedColormap

X_set, Y_set = X_train, Y_train

X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),

                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))

Xpred = np.array([X1.ravel(), X2.ravel()] + [np.repeat(0, X1.ravel().size) for _ in range(15)]).T

pred = classifier.predict(Xpred).reshape(X1.shape)

plt.contourf(X1, X2, pred,

             alpha = 0.75, cmap = ListedColormap(('red', 'green')))

plt.xlim(X1.min(), X1.max())

plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(Y_set)):

    plt.scatter(X_set[Y_set == j, 0], X_set[Y_set == j, 1],

                c = ListedColormap(('red', 'green'))(i), label = j)

plt.title('Kernel SVM (Training set)')

plt.xlabel('x')

plt.ylabel('y')

plt.legend()

plt.show()
X_set, Y_set = X_test, Y_test

X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),

                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))

Xpred = np.array([X1.ravel(), X2.ravel()] + [np.repeat(0, X1.ravel().size) for _ in range(15)]).T

pred = classifier.predict(Xpred).reshape(X1.shape)

plt.contourf(X1, X2, pred,

             alpha = 0.75, cmap = ListedColormap(('red', 'green')))

plt.xlim(X1.min(), X1.max())

plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(Y_set)):

    plt.scatter(X_set[Y_set == j, 0], X_set[Y_set == j, 1],

                c = ListedColormap(('red', 'green'))(i), label = j)

plt.title('Kernel SVM (Training set)')

plt.xlabel('x')

plt.ylabel('y')

plt.legend()

plt.show()