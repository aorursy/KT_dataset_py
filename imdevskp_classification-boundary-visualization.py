import numpy as np



from mlxtend.plotting import plot_decision_regions

import matplotlib.pyplot as plt



from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB 

from sklearn.ensemble import RandomForestClassifier
# Loading some example data

iris = datasets.load_iris()

X = iris.data[:, 2]

X = X[:, None]

y = iris.target



# Training a classifier

svm = SVC(C=0.5, kernel='linear')

svm.fit(X, y)



# Plotting decision regions

plot_decision_regions(X, y, clf=svm, legend=2)



# Adding axes annotations

plt.xlabel('sepal length [cm]')

plt.title('SVM on Iris')



plt.show()
from mlxtend.plotting import plot_decision_regions

import matplotlib.pyplot as plt

from sklearn import datasets

from sklearn.svm import SVC



# Loading some example data

iris = datasets.load_iris()

X = iris.data[:, [0,3]]

y = iris.target



# Training a classifier

svm = SVC(C=0.5, kernel='linear')

svm.fit(X, y)



# Plotting decision regions

plot_decision_regions(X, y, clf=svm, legend=2)



# Adding axes annotations

plt.xlabel('sepal length [cm]')

plt.ylabel('petal length [cm]')

plt.title('SVM on Iris')

plt.show()




# Initializing Classifiers

clf1 = LogisticRegression(random_state=1,

                          solver='newton-cg',

                          multi_class='multinomial')

clf2 = RandomForestClassifier(random_state=1, n_estimators=100)

clf3 = GaussianNB()

clf4 = SVC(gamma='auto')



# Loading some example data

iris = datasets.load_iris()

X = iris.data[:, [0,2]]

y = iris.target



import matplotlib.pyplot as plt

from mlxtend.plotting import plot_decision_regions

import matplotlib.gridspec as gridspec

import itertools

gs = gridspec.GridSpec(2, 2)



fig = plt.figure(figsize=(10,8))



labels = ['Logistic Regression', 'Random Forest', 'Naive Bayes', 'SVM']

for clf, lab, grd in zip([clf1, clf2, clf3, clf4],

                         labels,

                         itertools.product([0, 1], repeat=2)):



    clf.fit(X, y)

    ax = plt.subplot(gs[grd[0], grd[1]])

    fig = plot_decision_regions(X=X, y=y, clf=clf, legend=2)

    plt.title(lab)



plt.show()