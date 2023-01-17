# import library

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier



# import data

df = pd.read_csv("../input/pulsar_stars.csv")
display(df.shape)

display(df.info())

df.head()
display(df.corr())
# fig = plt.figure('Pulsar Stars', figsize = (15,20))

sns.pairplot(df)

# plt.tight_layout()

# plt.show()
# data selection

X = df.drop(['target_class'], axis = 1)

y = df['target_class']



# splitting data for train & test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)
# logistic regression model

log_reg = LogisticRegression(solver = 'liblinear')

log_reg.fit(X_train, y_train)



# scoring model

print('Accuracy = {}%'.format(round(log_reg.score(X_train, y_train) * 100, 2)))
# knn model

def nNeighbors():

    x = round(len(X_train) ** 0.5)

    if x % 2 == 0:

        return x + 1

    else:

        return x



knn = KNeighborsClassifier(n_neighbors = nNeighbors(), metric = 'minkowski', p = 2)

knn.fit(X_train, y_train)



# scoring model

print('Accuracy = {}%'.format(round(knn.score(X_train, y_train) * 100, 2)))
# kernel svm model

svm = SVC(kernel = 'rbf', gamma = 0.5)

svm.fit(X_train, y_train)



# scoring model

print('Accuracy = {}%'.format(round(svm.score(X_train, y_train) * 100, 2)))
# naive bayes model

gnb = GaussianNB()

gnb.fit(X_train, y_train)



# scoring model

print('Accuracy = {}%'.format(round(gnb.score(X_train, y_train) * 100, 2)))
# decision tree classifier

dct_clf = DecisionTreeClassifier(criterion = 'entropy')

dct_clf.fit(X_train, y_train)



# scoring model

print('Accuracy = {}%'.format(round(dct_clf.score(X_train, y_train) * 100, 2)))
# random forest model

rf_clf = RandomForestClassifier(n_estimators = 25, criterion = 'entropy')

rf_clf.fit(X_train, y_train)



# scoring model

print('Accuracy = {}%'.format(round(rf_clf.score(X, y) * 100, 2)))