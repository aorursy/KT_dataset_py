import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
ds = pd.read_csv('../input/heart.csv')
ds.info()
sns.heatmap(ds.isnull(), yticklabels = False, cbar=False, cmap='viridis')
ds.head(5)
sns.set_style('whitegrid')
sns.countplot(x='target', hue='sex', data=ds, palette='RdBu_r')
sns.countplot(x='target', hue='cp', data=ds)
ds['age'].plot.hist(bins=35)
ds.head(5)
ds['trestbps'].hist(bins=40, figsize=(10, 4))
plt.figure(figsize=(10, 8))

sns.boxplot(x='ca', y='age', data=ds)
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report, confusion_matrix
ds.head(5)
y = ds.iloc[:, 13].values.reshape(-1, 1)

X = ds.iloc[:, 0:13].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

X_train = sc_X.fit_transform(X_train)

X_test = sc_X.transform(X_test)
# Logistic Regression
model_forScaledFeatures = LogisticRegression()

model_forScaledFeatures.fit(X_train, y_train)

y_pred_forscaledfeatures = model_forScaledFeatures.predict(X_test)

print(classification_report(y_test, y_pred_forscaledfeatures))

print('\n')

print(confusion_matrix(y_test, y_pred_forscaledfeatures))
# K-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier
#Elbow method

error_rate = []



for i in range(1, 30):

    knn = KNeighborsClassifier(n_neighbors=i, metric='minkowski', p=2)

    knn.fit(X_train, y_train)

    pred_i = knn.predict(X_test)

    error_rate.append(np.mean(pred_i != y_test))
plt.figure(figsize=(12, 8))

plt.plot(range(1, 30), error_rate, color='blue', linestyle='dashed', marker='o', markerfacecolor='red', markersize=10)

plt.title('Error Rate v/s K value')

plt.xlabel('K value')

plt.ylabel('Error rate')

plt.show()
classifier_knn = KNeighborsClassifier(n_neighbors=11, metric='minkowski', p=2)
classifier_knn.fit(X_train, y_train)
ypred_from_knn = classifier_knn.predict(X_test)
print(classification_report(y_test, ypred_from_knn))

print('\n')

print(confusion_matrix(y_test, ypred_from_knn))
#SVM 
from sklearn.svm import SVC
model_SVC = SVC(kernel='linear', random_state=0)

model_SVC.fit(X_train, y_train)

ypred_from_svc = model_SVC.predict(X_test)

print(classification_report(y_test, ypred_from_svc))

print('\n')

print(confusion_matrix( y_test, ypred_from_svc))
#Kernel SVM (Gaussian Kernel)
model_SVM_Kernel = SVC(kernel='rbf', random_state=0)

model_SVM_Kernel.fit(X_train, y_train)

ypred_from_SVMKernel = model_SVM_Kernel.predict(X_test)

print(classification_report(y_test, ypred_from_SVMKernel))

print('\n')

print(confusion_matrix( y_test, ypred_from_SVMKernel))
#Decision-Tree
from sklearn.tree import DecisionTreeClassifier
model_tree = DecisionTreeClassifier()

model_tree.fit(X_train, y_train)

ypred_from_tree = model_tree.predict(X_test)

print(classification_report(y_test, ypred_from_tree))

print('\n')

print(confusion_matrix( y_test, ypred_from_tree))
#Random-Forest
from sklearn.ensemble import RandomForestClassifier
model_randomtree = RandomForestClassifier(n_estimators=200)

model_randomtree.fit(X_train, y_train)

ypred_from_randomtree = model_randomtree.predict(X_test)

print(classification_report(y_test, ypred_from_randomtree))

print('\n')

print(confusion_matrix( y_test, ypred_from_randomtree))