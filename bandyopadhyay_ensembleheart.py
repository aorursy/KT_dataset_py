# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os



abs_path = ''

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        abs_path = os.path.join(dirname, filename)

        print(abs_path)



# Any results you write to the current directory are saved as output.
record = np.genfromtxt(abs_path,delimiter=',')[1:]

Y = record[:,-1]

X = record[:,:-1 ]
'''Normalization'''



from sklearn import preprocessing



X = preprocessing.normalize(X, axis=0)
'''Data-Splitting'''



from sklearn.model_selection import train_test_split



X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33)



print(X_train.shape, X_test.shape)
'''Model and Prediction'''



from sklearn import metrics

from sklearn.ensemble import RandomForestClassifier



clf = RandomForestClassifier(n_estimators=100)

clf.fit(X_train, Y_train)

Y_pred = clf.predict(X_test)



print('Accuracy: ', metrics.accuracy_score(Y_test, Y_pred))
'''Ensemble Model and Prediction'''

from sklearn import svm

from xgboost import XGBClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.neural_network import MLPClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier, VotingClassifier

from sklearn.model_selection import cross_val_score, cross_val_predict



n_estimators = 100

# Range of classifiers experimented with

RF_clf = RandomForestClassifier(n_estimators=n_estimators)

XG_clf = XGBClassifier(n_estimators=n_estimators)

MLP_clf = MLPClassifier(hidden_layer_sizes=(100, 100), random_state=1, max_iter=1000)

LR_clf = LogisticRegression()

GNB_clf = GaussianNB()

SVM_clf = svm.SVC(kernel='rbf', probability=True)

KNN_clf = KNeighborsClassifier(n_neighbors=5)



E_clf = VotingClassifier(estimators=[('mlp', MLP_clf),  ('gnb', GNB_clf), ('svm', SVM_clf), ('rf', RF_clf)], voting='soft')



for clf, label in zip([LR_clf, RF_clf, GNB_clf, XG_clf, SVM_clf, MLP_clf, KNN_clf, E_clf], ['Logistic Regression', 'Random Forest', 'Naive Bayes', 'XG_boost', 'SVM', 'MLP', 'KNN', 'Ensemble']):

        scores = cross_val_score(clf, X, Y, cv=5, scoring='accuracy')

        print (scores)

        print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))