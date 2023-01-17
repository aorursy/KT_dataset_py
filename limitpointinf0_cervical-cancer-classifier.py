import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import preprocessing, neighbors, svm

from sklearn.model_selection import cross_val_score, train_test_split

import matplotlib.pyplot as plt

from keras.models import Sequential

from keras.layers import Dense

from sklearn.ensemble import RandomForestClassifier



df = pd.read_csv('../input/kag_risk_factors_cervical_cancer.csv')



df = df.replace('?',-1)

print(df.head())

X = np.array(df.drop('Biopsy',1))

X = preprocessing.scale(X)

y = np.array(df['Biopsy'])



# for i in range(1,20):

accuracy = []

x_range = []

for j in range(1000):

    x_range.append(j)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    clf = neighbors.KNeighborsClassifier(n_neighbors=1)

    clf.fit(X_train,y_train)

    acc = clf.score(X_test,y_test)

    accuracy.append(acc)

plt.title(str(1) + ' nearest neighbors')

plt.plot(x_range, accuracy)

plt.xlabel('Iteration')

plt.ylabel('Accuracy')

plt.show()

clf = neighbors.KNeighborsClassifier(n_neighbors=1)

clf.fit(X_train,y_train)



predictions = clf.predict(X_test)

print(predictions)

print(y_test)
accuracy = []

x_range = []

for j in range(1000):

    x_range.append(j)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    rf = RandomForestClassifier(n_estimators=100)

    rf.fit(X_train,y_train)

    acc = rf.score(X_test,y_test)

    accuracy.append(acc)

plt.title('Random Forest Classifier')

plt.plot(x_range, accuracy)

plt.xlabel('Iteration')

plt.ylabel('Accuracy')

plt.show()
#  for i in range(1,30):

#     accuracy = []

#     x_range = []

for j in range(1000):

    x_range.append(j)

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

    sv_clf = svm.SVC(degree=10)

    sv_clf.fit(X_train,y_train)

    acc = sv_clf.score(X_test,y_test)

    accuracy.append(acc)

plt.title('SVC degree ' + str(10))

plt.plot(x_range, accuracy)

plt.xlabel('Iteration')

plt.ylabel('Accuracy')

plt.show()