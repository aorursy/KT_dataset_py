

# Import data and modules

import pandas as pd

import numpy as np

from sklearn import datasets

%pylab inline

pylab.rcParams['figure.figsize'] = (10, 6)



iris = datasets.load_iris()



# We'll use the petal length and width only for this analysis

X = iris.data[:, [1, 2]]

y = iris.target



# Place the iris data into a pandas dataframe

iris_df = pd.DataFrame(iris.data[:, [2, 3]], columns=iris.feature_names[2:])



# View the first 5 rows of the data

print(iris_df.head())



# Print the unique labels of the dataset

print('\n' + 'The unique labels in this data are ' + str(np.unique(y)))
from sklearn.cross_validation import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3,

                                                random_state=0)



print('There are {} samples in the training set and {} samples in the test set'.format(

X_train.shape[0], X_test.shape[0]))

print()
from sklearn.preprocessing import StandardScaler



sc = StandardScaler()



sc.fit(X_train)



X_train_std = sc.transform(X_train)

X_test_std = sc.transform(X_test)



print('After standardizing our features, the first 5 rows of our data now look like this:\n')

print(pd.DataFrame(X_train_std, columns=iris_df.columns).head())
from matplotlib.colors import ListedColormap

import matplotlib.pyplot as plt



markers = ('s', 'x', 'o')

colors = ('red', 'blue', 'lightgreen')

cmap = ListedColormap(colors[:len(np.unique(y_test))])

for idx, cl in enumerate(np.unique(y)):

    plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],

               c=cmap(idx), marker=markers[idx], label=cl)
from sklearn.svm import SVC



svm = SVC(kernel='rbf', random_state=0, gamma=.10, C=1.0)

svm.fit(X_train_std, y_train)



print('The accuracy of the svm classifier on training data is {:.2f} out of 1'.format(svm.score(X_train_std, y_train)))



print('The accuracy of the svm classifier on test data is {:.2f} out of 1'.format(svm.score(X_test_std, y_test)))
import warnings





def versiontuple(v):

    return tuple(map(int, (v.split("."))))





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
from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier(n_neighbors=1, p=1, metric='minkowski')

knn.fit(X_train_std, y_train)



plot_decision_regions(X_train_std, y_train, knn)


knn = KNeighborsClassifier(n_neighbors=10, p=1, metric='minkowski')

knn.fit(X_train_std, y_train)



plot_decision_regions(X_train_std, y_train, knn)
from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier(n_neighbors=10, metric='manhattan')

knn.fit(X_train_std, y_train)



plot_decision_regions(X_train_std, y_train, knn)
from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier(n_neighbors=10, metric='minkowski')

knn.fit(X_train_std, y_train)



plot_decision_regions(X_train_std, y_train, knn)
from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier(n_neighbors=1, p=2, metric='minkowski')

knn.fit(X_train_std, y_train)



plot_decision_regions(X_train_std, y_train, knn)
plot_decision_regions(X_test_std, y_test, knn)
import xgboost as xgb



xgb_clf = xgb.XGBClassifier(max_depth = 6, n_estimators = 10)

xgb_clf = xgb_clf.fit(X_train_std, y_train)



print('The accuracy of the knn classifier is {:.2f} out of 1 on training data'.format(xgb_clf.score(X_train_std, y_train)))

print('The accuracy of the knn classifier is {:.2f} out of 1 on test data'.format(xgb_clf.score(X_test_std, y_test)))
plot_decision_regions(X_train_std, y_train, xgb_clf)