# This Python 3 environment comes with many helpful analytics libraries installedimport numpy as np

import pandas

from sklearn.preprocessing import scale

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.neighbors import KNeighborsClassifier
data = pandas.read_csv('/UCL Wine/wine.data')

#data.head()
X = np.array(data.iloc[:, 1:14])

y = np.array(data.iloc[:, 0])

X = scale(X)
kf = KFold(n_splits=5, random_state=42, shuffle=True)

kf.get_n_splits(X)

accuracies = []

for x in range(1, 50):

    clf = KNeighborsClassifier(n_neighbors=x)

    scores = cross_val_score(clf, X, y, cv=kf, scoring='accuracy')

    #print(scores)

    accuracies.insert(x, scores.mean())

    print("Accuracy %2d: %0.2f (+/- %0.2f)" % (x, scores.mean(), scores.std() * 2))