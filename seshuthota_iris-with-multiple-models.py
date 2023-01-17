# Imports



# pandas

import pandas as pd

from pandas import Series,DataFrame



# numpy, matplotlib, seaborn

import numpy as np

import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap

import seaborn as sns

sns.set_style('whitegrid')

%matplotlib inline



# sklearn

from sklearn import datasets

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.tree import DecisionTreeClassifier

from sklearn.cross_validation import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score
iris = pd.read_csv("../input/Iris.csv")

#pairplot in seaborn to see how featurs are related to each other

sns.pairplot(iris.drop("Id", axis=1), hue="Species", size=2)
X = iris.loc[:,['PetalLengthCm','PetalWidthCm']]

y = iris["Species"]

def get_series_ids(x):#converting string to integer values

    '''Function returns a pandas series consisting of ids, 

       corresponding to objects in input pandas series x

       Example: 

       get_series_ids(pd.Series(['a','a','b','b','c'])) 

       returns Series([0,0,1,1,2], dtype=int)'''



    values = np.unique(x)

    values2nums = dict(zip(values,range(len(values))))

    return x.replace(values2nums)

y = get_series_ids(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

sc = StandardScaler()

sc.fit(X_train)

X_train_std = sc.transform(X_train)

X_test_std = sc.transform(X_test)
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
ppn = Perceptron(n_iter=40, eta0=0.1, random_state=0)

ppn.fit(X_train_std, y_train)

y1 = ppn.predict(X_test_std)

print('Misclassified samples: %d' % (y_test != y1).sum())

print('Accuracy: %.2f' % accuracy_score(y_test, y1))

X_combined_std = np.vstack((X_train_std, X_test_std))

y_combined = np.hstack((y_train, y_test))

plot_decision_regions(X=X_combined_std,y=y_combined,classifier=ppn,test_idx=range(105,150))

plt.xlabel('petal length [standardized]')

plt.ylabel('petal width [standardized]')

plt.legend(loc='upper left')

plt.show()
lr = LogisticRegression(C=1000.0, random_state=0)

lr.fit(X_train_std, y_train)

y2 = lr.predict(X_test_std)

print('Misclassified samples: %d' % (y_test != y2).sum())

print('Accuracy: %.2f' % accuracy_score(y_test, y2))

plot_decision_regions(X=X_combined_std,y=y_combined,classifier=lr,test_idx=range(105,150))

plt.xlabel('petal length [standardized]')

plt.ylabel('petal width [standardized]')

plt.legend(loc='upper left')

plt.show()
svm = SVC(kernel='linear', C=1.0, random_state=0)

svm.fit(X_train_std, y_train)

y3 = svm.predict(X_test_std)

print('Misclassified samples: %d' % (y_test != y3).sum())

print('Accuracy: %.2f' % accuracy_score(y_test, y3))

plot_decision_regions(X=X_combined_std,y=y_combined,classifier=svm,test_idx=range(105,150))

plt.xlabel('petal length [standardized]')

plt.ylabel('petal width [standardized]')

plt.legend(loc='upper left')

plt.show()
svmk = SVC(kernel='rbf', random_state=0, gamma=0.2, C=1.0)

svmk.fit(X_train_std, y_train)

y4 = svmk.predict(X_test_std)

print('Misclassified samples: %d' % (y_test != y4).sum())

print('Accuracy: %.2f' % accuracy_score(y_test, y4))

plot_decision_regions(X=X_combined_std,y=y_combined,classifier=svmk,test_idx=range(105,150))

plt.xlabel('petal length [standardized]')

plt.ylabel('petal width [standardized]')

plt.legend(loc='upper left')

plt.show()
svmkg = SVC(kernel='rbf', random_state=0, gamma=100.0, C=1.0)

svmkg.fit(X_train_std, y_train)

y5 = svmkg.predict(X_test_std)

print('Misclassified samples: %d' % (y_test != y5).sum())

print('Accuracy: %.2f' % accuracy_score(y_test, y5))

plot_decision_regions(X=X_combined_std,y=y_combined,classifier=svmkg,test_idx=range(105,150))

plt.xlabel('petal length [standardized]')

plt.ylabel('petal width [standardized]')

plt.legend(loc='upper left')

plt.show()
tree = DecisionTreeClassifier(criterion='entropy',max_depth=3, random_state=0)

tree.fit(X_train, y_train)

X_combined = np.vstack((X_train, X_test))

y_combined = np.hstack((y_train, y_test))

y6 = tree.predict(X_test)

print('Misclassified samples: %d' % (y_test != y6).sum())

print('Accuracy: %.2f' % accuracy_score(y_test, y6))

plot_decision_regions(X=X_combined,y=y_combined,classifier=tree,test_idx=range(105,150))

plt.xlabel('petal length [standardized]')

plt.ylabel('petal width [standardized]')

plt.legend(loc='upper left')

plt.show()
forest = RandomForestClassifier(criterion='entropy',n_estimators=10,random_state=1,n_jobs=2)

forest.fit(X_train, y_train)

y7 = tree.predict(X_test)

print('Misclassified samples: %d' % (y_test != y7).sum())

print('Accuracy: %.2f' % accuracy_score(y_test, y7))

plot_decision_regions(X=X_combined,y=y_combined,classifier=forest,test_idx=range(105,150))

plt.xlabel('petal length [standardized]')

plt.ylabel('petal width [standardized]')

plt.legend(loc='upper left')

plt.show()
knn = KNeighborsClassifier(n_neighbors=5, p=2,metric='minkowski')

knn.fit(X_train_std, y_train)

y8 = knn.predict(X_test_std)

print('Misclassified samples: %d' % (y_test != y8).sum())

print('Accuracy: %.2f' % accuracy_score(y_test, y8))

plot_decision_regions(X=X_combined_std,y=y_combined,classifier=knn,test_idx=range(105,150))

plt.xlabel('petal length [standardized]')

plt.ylabel('petal width [standardized]')

plt.legend(loc='upper left')

plt.show()