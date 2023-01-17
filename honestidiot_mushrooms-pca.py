import pandas as pd

from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.svm import SVC

from sklearn.decomposition import PCA

import time
mushrooms = pd.read_csv("../input/mushrooms.csv")
le = LabelEncoder()

dup1 = mushrooms.copy()

for i in mushrooms.columns:

    dup1[i] = le.fit_transform(mushrooms[i])
dup1.drop('veil-type', axis=1, inplace=True)
X = dup1.drop('class', axis=1)

Y = dup1['class']
sc = StandardScaler()

X = sc.fit_transform(X)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
start = time.time()

clf = SVC(kernel='rbf')

clf.fit(x_train, y_train)

print(clf.score(x_test, y_test))

stop = time.time()

print(stop-start)
le = LabelEncoder()

dup2 = mushrooms.copy()

for i in mushrooms.columns:

    dup2[i] = le.fit_transform(mushrooms[i])
dup2.drop('veil-type', axis=1, inplace=True)
X = dup1.drop('class', axis=1)

Y = dup1['class']
sc = StandardScaler()

X = sc.fit_transform(X)
pca = PCA()

pca.fit(X)

v = pca.explained_variance_



tot = v.sum()

sum = 0

for i in range(v.shape[0]):

    sum = sum + v[i]/tot

    if sum > 0.95:

        break

        

pca.n_components = i

X = pca.transform(X)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
start = time.time()

clf = SVC(kernel='rbf')

clf.fit(x_train, y_train)

print(clf.score(x_test, y_test))

stop = time.time()

print(stop-start)