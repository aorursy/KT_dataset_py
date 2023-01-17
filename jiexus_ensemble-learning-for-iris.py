import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import VotingClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

iris = pd.read_csv("../input/Iris.csv")



iris.describe()
iris.head(5)
iris['Species'].value_counts()
X = iris.iloc[:, 0:3]

y = iris.iloc[:, 4]

y.head()
from matplotlib.colors import ListedColormap

import matplotlib.pyplot as plt



markers = ('s', 'x', 'o')

colors = ('red', 'blue', 'lightgreen')

cmap = ListedColormap(colors[:len(np.unique(y_test))])

for idx, cl in enumerate(np.unique(y)):

    plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],

               c=cmap(idx), marker=markers[idx], label=cl)
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=0)

print('There are {} samples in the training set and {} samples in the test set'.format(

X_train.shape[0], X_test.shape[0]))

print()



from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

sc.fit(X_train)

X_train_std = sc.transform(X_train)

X_test_std = sc.transform(X_test)
log_clf = LogisticRegression()

#LogisticRegression(penalty=’l2’, dual=False, tol=0.0001, C=1.0, fit_intercept=True, 

#                   intercept_scaling=1, class_weight=None, random_state=None, 

#                   solver=’liblinear’, max_iter=100, 

#                   multi_class=’ovr’, verbose=0, 

#                   warm_start=False, n_jobs=1)

rnd_clf = RandomForestClassifier()

svm_clf = SVC()

#(C=1.0, kernel=’rbf’, degree=3, gamma=’auto’, coef0=0.0, 

# shrinking=True, probability=False, tol=0.001, cache_size=200, 

# class_weight=None, verbose=False, max_iter=-1, 

# decision_function_shape=’ovr’, random_state=None)

knn_clf = KNeighborsClassifier()

#n_neighbors=5, weights=’uniform’, algorithm=’auto’, 

#                               leaf_size=30, p=2, metric=’minkowski’,

#                               metric_params=None, n_jobs=1, **kwargs)



voting_clf = VotingClassifier(

    estimators = [('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf), ('knn', knn_clf)],

    voting = 'hard')

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):



    # setup marker generator and color map

    markers = ('s', 'x', 'o', '^', 'v')

    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')

    cmap = ListedColormap(colors[:len(np.unique(y))])



    # plot the decision surface

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1

    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),

                           np.arange(x2_min, x2_max, resolution))

    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)

    Z = Z.reshape(xx1.shape)

    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)

    plt.xlim(xx1.min(), xx1.max())

    plt.ylim(xx2.min(), xx2.max())



    for idx, cl in enumerate(np.unique(y)):

        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],

                    alpha=0.8, c=cmap(idx),

                    marker=markers[idx], label=cl)
plot_decision_regions(X_test_std, y_test, svm)