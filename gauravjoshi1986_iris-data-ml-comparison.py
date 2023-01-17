

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.datasets import load_iris

# import some data to play with

iris = load_iris()

#print(iris)

X = iris.data[:,2:]  # we only take the first two features.

y = iris.target
# devide the dataset into training and test data

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

sc.fit(X_train)

X_train_std = sc.transform(X_train)

X_test_std = sc.transform(X_test)
from sklearn.linear_model import Perceptron

ppn = Perceptron(n_iter=40, eta0=0.1, random_state=0)

ppn.fit(X_train_std, y_train)
y_pred = ppn.predict(X_test_std)



from sklearn.metrics import accuracy_score

print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
from matplotlib.colors import ListedColormap

import matplotlib.pyplot as plt

def plot_decision_regions(X, y, classifier,test_idx=None, resolution=0.02):

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

    # plot all samples

    X_test, y_test = X[test_idx, :], y[test_idx]

    

    for idx, cl in enumerate(np.unique(y)):

        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],alpha=0.8, c=cmap(idx),marker=markers[idx], label=cl)

        

    # highlight test samples

    if test_idx:

        X_test, y_test = X[test_idx, :], y[test_idx]

        plt.scatter(X_test[:, 0], X_test[:, 1], c='',alpha=1.0, linewidth=1, marker='o',s=55, label='test set')
X_combined_std = np.vstack((X_train_std, X_test_std))

y_combined = np.hstack((y_train, y_test))

plot_decision_regions(X=X_combined_std,y=y_combined,classifier=ppn,test_idx=range(105,150))



plt.xlabel('petal length [standardized]')

plt.ylabel('petal width [standardized]')

plt.legend(loc='upper left')



plt.show()
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(C=1000.0, random_state=0)

lr.fit(X_train_std, y_train)



#print (lr.coef_)
y_pred = lr.predict(X_test_std)



from sklearn.metrics import accuracy_score

print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
plot_decision_regions(X_combined_std,y_combined, classifier=lr,test_idx=range(105,150))

plt.xlabel('petal length [standardized]')

plt.ylabel('petal width [standardized]')

plt.legend(loc='upper left')

plt.show()
lr.predict_proba(X_test_std[0,:])
weights, params = [], []

for c in np.arange(-5, 5):

    lr = LogisticRegression(penalty = 'l2', C=10**c, random_state=0)

    lr.fit(X_train_std, y_train)

    weights.append(lr.coef_[1])

    params.append(10**c)

    

weights = np.array(weights)



plt.plot(params, weights[:, 0],label='petal length')

plt.plot(params, weights[:, 1], linestyle='--',label='petal width')

plt.ylabel('weight coefficient')

plt.xlabel('C')

plt.legend(loc='upper left')

plt.xscale('log')

plt.show()
from sklearn.svm import SVC

svm = SVC(kernel='rbf', random_state=0, gamma=.2, C=1.0)

svm.fit(X_train_std, y_train)
y_pred = svm.predict(X_test_std)



from sklearn.metrics import accuracy_score

print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
plot_decision_regions(X_combined_std,y_combined, classifier=svm,test_idx=range(105,150))

plt.xlabel('petal length [standardized]')

plt.ylabel('petal width [standardized]')

plt.legend(loc='upper left')

plt.show()
from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(criterion='entropy',n_estimators=20,random_state=1,n_jobs=2, oob_score=True)

forest.fit(X_train, y_train)
y_pred = forest.predict(X_test)



from sklearn.metrics import accuracy_score

print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
X_combined = np.vstack((X_train, X_test))

y_combined = np.hstack((y_train, y_test))



plot_decision_regions(X_combined, y_combined,classifier=forest, test_idx=range(105,150))

plt.xlabel('petal length')

plt.ylabel('petal width')

plt.legend(loc='upper left')

plt.show()
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5, p=2,metric='minkowski')

knn.fit(X_train_std, y_train)
y_pred = knn.predict(X_test_std)



from sklearn.metrics import accuracy_score

print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
plot_decision_regions(X_combined_std, y_combined,classifier=knn, test_idx=range(105,150))

plt.xlabel('petal length [standardized]')

plt.ylabel('petal width [standardized]')

plt.show()