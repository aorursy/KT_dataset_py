import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
datasets = pd.read_csv('../input/datasetknn/DataSetKNN.csv')
datasets.head()
sns.pairplot(datasets, hue='TARGET CLASS')
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(datasets.drop('TARGET CLASS', axis=1))
scaled_features = scaler.transform(datasets.drop('TARGET CLASS', axis=1))
datasets_feat = pd.DataFrame(scaled_features, columns = datasets.columns[:-1])

datasets_feat.head()
from sklearn.model_selection import train_test_split

X = datasets_feat

y = datasets['TARGET CLASS']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 1)
knn.fit(X_train, y_train)
pred = knn.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test, pred))

print(classification_report(y_test, pred))
import numpy as np

error_rate = []

for i in range(1,40):

    knn = KNeighborsClassifier(n_neighbors = i)

    knn.fit(X_train, y_train)

    pred_i = knn.predict(X_test)

    error_rate.append(np.mean(pred_i != y_test))
plt.figure(figsize=(15,6))

plt.plot(range(1,40),error_rate,color='blue',linestyle='dashed',marker='o',

markerfacecolor='red', markersize='10')

plt.xlabel('no. of K')

plt.ylabel('Error Rate')
knn = KNeighborsClassifier(n_neighbors = 31)

knn.fit(X_train, y_train)

pred = knn.predict(X_test)

print(confusion_matrix(y_test, pred))

print(classification_report(y_test, pred))