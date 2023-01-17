import numpy as np

import matplotlib.pyplot as plt

from sklearn import datasets, neighbors
from mlxtend.plotting import plot_decision_regions

import pandas as pd
def knn_comparision(data, k):

    X = data[['x1','x2']].values

    y = data['y'].astype(int).values

    clf = neighbors.KNeighborsClassifier(n_neighbors=k)

    clf.fit(X, y)



    # Plotting decision regions

    plot_decision_regions(X, y, clf=clf, legend=2)



    # Adding axes annotations

    plt.xlabel('X1')

    plt.ylabel('X2')

    plt.title('Knn with K='+ str(k))

    plt.show()
data = pd.read_csv('/kaggle/input/knndemo/6.overlap.csv', names=['x1', 'x2', 'y'])

for i in [1, 5, 15, 30, 45]:

    knn_comparision(data, i)
data = pd.read_csv('/kaggle/input/knndemo/1.ushape.csv', names=['x1', 'x2', 'y'])

print(data.head(3))

for i in [1, 5, 15, 30, 45]:

    knn_comparision(data, i)
data = pd.read_csv('/kaggle/input/knndemo/2.concerticcir1.csv', names=['x1', 'x2', 'y'])

print(data.head(3))

for i in [1, 5, 15, 30, 45]:

    knn_comparision(data, i)
data = pd.read_csv('/kaggle/input/knndemo/3.concertriccir2.csv', names=['x1', 'x2', 'y'])

print(data.head(3))

for i in [1, 5, 15, 30, 45]:

    knn_comparision(data, i)
data = pd.read_csv('/kaggle/input/knndemo/4.linearsep.csv', names=['x1', 'x2', 'y'])

print(data.head(3))

for i in [1, 5, 15, 30, 45]:

    knn_comparision(data, i)
data = pd.read_csv('/kaggle/input/knndemo/5.outlier.csv', names=['x1', 'x2', 'y'])

print(data.head(3))

for i in [1, 5, 15, 30, 45]:

    knn_comparision(data, i)
data = pd.read_csv('/kaggle/input/knndemo/7.xor.csv', names=['x1', 'x2', 'y'])

print(data.head(3))

for i in [1, 5, 15, 30, 45]:

    knn_comparision(data, i)
data = pd.read_csv('/kaggle/input/knndemo/8.twospirals.csv', names=['x1', 'x2', 'y'])

print(data.head(3))

for i in [1, 5, 15, 30, 45]:

    knn_comparision(data, i)
data = pd.read_csv('/kaggle/input/knndemo/9.random.csv', names=['x1', 'x2', 'y'])

print(data.head(3))

for i in [1, 5, 15, 30, 45]:

    knn_comparision(data, i)