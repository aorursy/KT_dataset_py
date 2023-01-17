import itertools

import numpy as np

import matplotlib.pyplot as plt

from matplotlib.ticker import NullFormatter

import pandas as pd

import matplotlib.ticker as ticker

from sklearn import preprocessing

%matplotlib inline
df = pd.read_csv("../input/telecustsclasses/teleCust1000t.csv")
df.head()
df['custcat'].value_counts()
df.hist(column = 'income', bins = 50)
df.columns
X = df[['region', 'tenure', 'age', 'marital', 'address', 'income', 'ed', 'employ', 'retire', 'gender', 'reside']].values

X[0:5]
y = df['custcat'].values

y[0:5]
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
X[0:5]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = 0.2, random_state = 4)

print('Train set:', X_train.shape, y_train.shape)

print('Test set:', X_test.shape, y_test.shape)
from sklearn.neighbors import KNeighborsClassifier
k = 4

neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train, y_train)

neigh
yhat = neigh.predict(X_test)

yhat[0:5]
y[0:5]
import sklearn.metrics as metrics

Ks = 10

mean_acc = np.zeros((Ks-1))

std_acc = np.zeros((Ks-1))

ConfusionMx = [];

for n in range(1, Ks):

    

    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train, y_train)

    yhat = neigh.predict(X_test)

    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)

    std_acc[n-1] = np.std(yhat == y_test)/np.sqrt(yhat.shape[0])

    

print(mean_acc)

print(std_acc)
plt.plot(range(1, Ks), mean_acc, 'g')

plt.fill_between(range(1, Ks), mean_acc -1 * std_acc, mean_acc + 1 *std_acc, alpha = 0.10)

plt.legend(('Accuracy', '+/- 3xstd'))

plt.ylabel('Accuracy')

plt.xlabel('Number of Neighbors (K)')

plt.tight_layout()

plt.show()
print("The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1)