# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

y_train = train.label.values.ravel()

X_train = train.values[:,1:]

X_test = test.values
def show(array):

    array = array.reshape(28,28)

    plt.figure(figsize=(3,3))

    plt.imshow(array, cmap='gray', interpolation='none')

    plt.show()

    

def plot_gallery(images, n_row=2, n_col=5):

    h = w = 28

    """Helper function to plot a gallery of portraits"""

    plt.figure(figsize=(2 * n_col, 2 * n_row))

    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)

    for i in range(n_row * n_col):

        plt.subplot(n_row, n_col, i + 1)

        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray, interpolation='none')

        plt.xticks(())

        plt.yticks(())
def normalized_data():

    X = np.concatenate((X_train, X_test))

    x_test = X_test - X.mean(0)

    x_train = X_train - X.mean(0)

    x_test /= x_test.std(1).reshape((-1,1))

    x_train /= x_train.std(1).reshape((-1,1))

    return x_train, x_test
from sklearn.decomposition import PCA, RandomizedPCA



def components(n_components, random=False, show_components=True):

    x_train, x_test = normalized_data()

    pca = PCA(n_components=n_components)

    if random:

        pca = RandomPCA(n_components=n_components)

    # components

    pca.fit(x_train)

    components = pca.components_

    if show_components:

        plot_gallery(components, n_col=5, n_row=int(n_components / 5))

    x_train = pca.transform(x_train)

    x_test = pca.transform(x_test)

    return x_train, x_test
from sklearn.grid_search import GridSearchCV

from sklearn.svm import SVC



def testCLF(clf, parameters, X):

    search = GridSearchCV(clf, parameters)

    search.fit(X, y_train)

    print(search.score)

    return search.best_estimator_
clf = SVC()

parameters = {

    "C" : [0.5, 1.0],

    "kernel" : ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed' ],

    "degree" : [2, 3]

}

X = components(10)[0]

testCLF(clf, parameters, X)