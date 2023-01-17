import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

data = pd.read_csv('../input/abalone/abalone.data.csv')
data.head()

X = data.iloc[:, 1:]
y = data.iloc[:, 0]
y = pd.Series(pd.Categorical(y).codes)
scalerX = StandardScaler()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=1)

X_train_scaled = scalerX.fit_transform(X_train)
X_test_scaled = scalerX.fit_transform(X_test)
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

def evaluate(y_pred, y_true):
    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)

    print("model confusion matrix\n", cm)
    print("model accuracy: ", acc)

classifier = KNeighborsClassifier(n_neighbors = 3)
classifier.fit(X_train_scaled, y_train)

y_pred = classifier.predict(X_test)

evaluate(y_pred, y_test)
# perform PCA to improve performance

from sklearn.decomposition import PCA
X_train_reduce = PCA(n_components=2).fit_transform(X_train_scaled)
X_test_reduced = PCA(n_components=2).fit_transform(X_test_scaled)

print(X_train_reduce.shape)
print(X_test_reduced.shape)
classifier.fit(X_train_reduce, y_train)

y_pred = classifier.predict(X_test_reduced)

evaluate(y_pred, y_test)
# pip install neupy
from neupy import algorithms

lvqnet = algorithms.LVQ(n_inputs=X_train_scaled.shape[0], n_classes=3)

lvqnet.train(X_train_scaled, y_train, epochs=100)

y_pred = lvqnet.predict(X_test_scaled)

evaluate(y_pred, y_test)