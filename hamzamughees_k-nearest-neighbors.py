# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from collections import Counter

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from mlxtend.plotting import plot_confusion_matrix

from matplotlib.colors import ListedColormap



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
class Quick_Sort:

    def __init__(self):

        self.lst = []

        self.inds = []

    

    def sort(self, lst):

        self.lst = lst

        self.inds = [i for i in range(len(self.lst))]

        self._sort(self.lst, 0, len(self.lst)-1)



    def _sort(self, lst, start, end):

        if start >= end: return

        index = self._partition(lst, start, end)

        self._sort(lst, start, index-1)

        self._sort(lst, index+1, end)



    def _partition(self, lst, start, end):

        piv_ind = start

        piv_val = lst[end]

        for i in range(start, end):

            if lst[i] < piv_val:

                self._swap(i, piv_ind)

                piv_ind += 1

        self._swap(piv_ind, end)

        return piv_ind



    def _swap(self, i, j):

        self.lst[i], self.lst[j] = self.lst[j], self.lst[i]

        self.inds[i], self.inds[j] = self.inds[j], self.inds[i]
class KNN:

    def __init__(self, n_neighbors=5):

        self.n_neighbors = n_neighbors

    

    def _euclidean_distance(self, p, q):

        return np.sqrt(np.sum((q-p)**2))

    

    def fit(self, X, y):

        self.X = X

        self.y = y



    def predict(self, X):

        return np.array([self._predict(p) for p in X])



    def _predict(self, p):

        distances = [self._euclidean_distance(p, q) for q in self.X]

        qs = Quick_Sort()

        qs.sort(distances)

        n_nearest_neighbors_indeces = qs.inds[:self.n_neighbors]

        n_nearest_neighbors_y = [self.y[i] for i in n_nearest_neighbors_indeces]

        return Counter(n_nearest_neighbors_y).most_common(1)[0][0]
dataset = pd.read_csv('/kaggle/input/iris/Iris.csv')

X = dataset.iloc[:, 1:-1].values

y = dataset.iloc[:, -1].values
le = LabelEncoder()

y = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
classifier = KNN()

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))
cm = confusion_matrix(y_test, y_pred)

a, b = plot_confusion_matrix(conf_mat=cm, figsize=(2, 2))
cmap = ListedColormap(['#ff0000', '#00ff00', '#0000ff'])

plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cmap, edgecolor='k', s=20)

plt.title('True')

plt.show()
cmap = ListedColormap(['#ff0000', '#00ff00', '#0000ff'])

plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap=cmap, edgecolor='k', s=20)

plt.title('Predicted')

plt.show()
sizes = np.array([cm[0][0]+cm[1][1]+cm[2][2], cm[0][1]+cm[0][2]+cm[1][0]+cm[1][2]+cm[2][0]+cm[2][1]])

labels = 'Correct', 'Incorrect'

colours = ['blue', 'red']



plt.pie(sizes, labels=labels, colors=colours, shadow=True)

plt.show()