# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv('../input/breast-cancer-wisconsin-data/data.csv')
X = dataset.iloc[:, 2:4].values
y = dataset.iloc[:, 1].values
dataset.head()
# As y contains text, we need to encode it.
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
# Splitting dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
# Applying Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# Trainig the Model
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print('Accuracy = '+str(accuracy_score(y_test, y_pred)))

import seaborn as sns
plt.subplots(figsize=(5,5))
ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax);

ax.set_xlabel('Prediction');ax.set_ylabel('Label'); 
ax.set_title('Confusion Matrix'); 
def Label(val):
    if val==0:
        return 'Malignant'
    else:
        return 'Benign'
from matplotlib.colors import ListedColormap
plt.style.use('fivethirtyeight')
X_set, y_set = sc.inverse_transform(X_test), y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 0.15),
                     np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 0.25))
plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.3, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = Label(j), alpha = 0.7, s = 50)
plt.title('Logistic Regression')
plt.xlabel('Mean-Texture')
plt.ylabel('Mean-Radius')
plt.axis([5,25,0,50])
plt.legend(loc = 'upper left')
plt.show()
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 35, metric = 'minkowski', p = 2)
# I increased the value of K (Number of neighbours) as the model was overfitting with less number of neighbours.
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print('Accuracy = '+str(accuracy_score(y_test, y_pred)))

import seaborn as sns
plt.subplots(figsize=(5,5))
ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax);

ax.set_xlabel('Prediction');ax.set_ylabel('Label'); 
ax.set_title('Confusion Matrix'); 
def Label(val):
    if val==0:
        return 'Malignant'
    else:
        return 'Benign'
from matplotlib.colors import ListedColormap
plt.style.use('fivethirtyeight')
X_set, y_set = sc.inverse_transform(X_test), y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 0.15),
                     np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 0.25))
plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.3, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = Label(j), alpha = 0.7, s = 50)
plt.title('K-Nearest Neighbor')
plt.xlabel('Mean-Texture')
plt.ylabel('Mean-Radius')
plt.axis([5,25,0,50])
plt.legend(loc = 'upper left')
plt.show()
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print('Accuracy = '+str(accuracy_score(y_test, y_pred)))

import seaborn as sns
plt.subplots(figsize=(5,5))
ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax);

ax.set_xlabel('Prediction');ax.set_ylabel('Label'); 
ax.set_title('Confusion Matrix'); 
def Label(val):
    if val==0:
        return 'Malignant'
    else:
        return 'Benign'
from matplotlib.colors import ListedColormap
plt.style.use('fivethirtyeight')
X_set, y_set = sc.inverse_transform(X_test), y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 0.15),
                     np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 0.25))
plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.3, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = Label(j), alpha = 0.7, s = 50)
plt.title('Support Vector Machine')
plt.xlabel('Mean-Texture')
plt.ylabel('Mean-Radius')
plt.axis([5,25,0,50])
plt.legend(loc = 'upper left')
plt.show()
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0, C=0.5, gamma=0.5)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print('Accuracy = '+str(accuracy_score(y_test, y_pred)))

import seaborn as sns
plt.subplots(figsize=(5,5))
ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax);

ax.set_xlabel('Prediction');ax.set_ylabel('Label'); 
ax.set_title('Confusion Matrix'); 
def Label(val):
    if val==0:
        return 'Malignant'
    else:
        return 'Benign'
from matplotlib.colors import ListedColormap
plt.style.use('fivethirtyeight')
X_set, y_set = sc.inverse_transform(X_test), y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 0.15),
                     np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 0.25))
plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.3, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = Label(j), alpha = 0.7, s = 50)
plt.title('Kernel SVM')
plt.xlabel('Mean-Texture')
plt.ylabel('Mean-Radius')
plt.axis([5,25,0,50])
plt.legend(loc = 'upper left')
plt.show()
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print('Accuracy = '+str(accuracy_score(y_test, y_pred)))

import seaborn as sns
plt.subplots(figsize=(5,5))
ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax);

ax.set_xlabel('Prediction');ax.set_ylabel('Label'); 
ax.set_title('Confusion Matrix'); 
def Label(val):
    if val==0:
        return 'Malignant'
    else:
        return 'Benign'
from matplotlib.colors import ListedColormap
plt.style.use('fivethirtyeight')
X_set, y_set = sc.inverse_transform(X_test), y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 0.15),
                     np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 0.25))
plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.3, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = Label(j), alpha = 0.7, s = 50)
plt.title('Naive Bayes')
plt.xlabel('Mean-Texture')
plt.ylabel('Mean-Radius')
plt.axis([5,25,0,50])
plt.legend(loc = 'upper left')
plt.show()
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0, max_depth=3, min_samples_split=0.8)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print('Accuracy = '+str(accuracy_score(y_test, y_pred)))

import seaborn as sns
plt.subplots(figsize=(5,5))
ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax);

ax.set_xlabel('Prediction');ax.set_ylabel('Label'); 
ax.set_title('Confusion Matrix'); 
def Label(val):
    if val==0:
        return 'Malignant'
    else:
        return 'Benign'
from matplotlib.colors import ListedColormap
plt.style.use('fivethirtyeight')
X_set, y_set = sc.inverse_transform(X_test), y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 0.15),
                     np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 0.25))
plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.3, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = Label(j), alpha = 0.7, s = 50)
plt.title('Decision Tree Classification')
plt.xlabel('Mean-Texture')
plt.ylabel('Mean-Radius')
plt.axis([5,25,0,50])
plt.legend(loc = 'upper left')
plt.show()
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 20, criterion = 'entropy', random_state = 0, max_depth=5)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print('Accuracy = '+str(accuracy_score(y_test, y_pred)))

import seaborn as sns
plt.subplots(figsize=(5,5))
ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax);

ax.set_xlabel('Prediction');ax.set_ylabel('Label'); 
ax.set_title('Confusion Matrix'); 
def Label(val):
    if val==0:
        return 'Malignant'
    else:
        return 'Benign'
from matplotlib.colors import ListedColormap
plt.style.use('fivethirtyeight')
X_set, y_set = sc.inverse_transform(X_test), y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 0.15),
                     np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 0.25))
plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.3, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = Label(j), alpha = 0.7, s = 50)
plt.title('Random Forest Classification')
plt.xlabel('Mean-Texture')
plt.ylabel('Mean-Radius')
plt.axis([5,25,0,50])
plt.legend(loc = 'upper left')
plt.show()