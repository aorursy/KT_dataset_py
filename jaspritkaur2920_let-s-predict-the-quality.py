import numpy as np

import pandas as pd

import seaborn as sns

from collections import Counter

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.linear_model import SGDClassifier

from sklearn.metrics import confusion_matrix, classification_report

from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

%matplotlib inline
wine = pd.read_csv("../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv")
#Let's check how the data is distributed

wine.head()
wine.info()
sns.countplot(x='quality',data=wine)
fig = plt.figure(figsize = (10,6))

sns.barplot(x='quality', y='fixed acidity', data=wine)
fig = plt.figure(figsize = (10,6))

sns.barplot(x='quality', y='volatile acidity', data=wine)
fig = plt.figure(figsize = (10,6))

sns.barplot(x='quality', y='citric acid', data=wine)
fig = plt.figure(figsize = (10,6))

sns.barplot(x='quality', y='residual sugar', data=wine)
fig = plt.figure(figsize = (10,6))

sns.barplot(x='quality', y='chlorides', data=wine)
fig = plt.figure(figsize = (10,6))

sns.barplot(x='quality', y='free sulfur dioxide', data=wine)
fig = plt.figure(figsize = (10,6))

sns.barplot(x='quality', y='total sulfur dioxide', data=wine)
fig = plt.figure(figsize = (10,6))

sns.barplot(x='quality', y='sulphates', data=wine)
fig = plt.figure(figsize = (10,6))

sns.barplot(x='quality', y='alcohol', data=wine)
reviews = []

for i in wine['quality']:

    if i >= 1 and i <= 3:

        reviews.append('1')

    elif i >= 4 and i <= 7:

        reviews.append('2')

    elif i >= 8 and i <= 10:

        reviews.append('3')

wine['Reviews'] = reviews
wine.columns
wine['Reviews'].unique()
Counter(wine['Reviews'])
X = wine.iloc[:,:11]

y = wine['Reviews']
X.head()
y.head()
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X = sc.fit_transform(X)
print(X)
from sklearn.decomposition import PCA

pca = PCA()

X_pca = pca.fit_transform(X)
plt.figure(figsize=(5,5))

plt.plot(np.cumsum(pca.explained_variance_ratio_), 'ro-')

plt.grid()
#As per the graph, we can see that 8 principal components attribute for 90% of variation in the data. 

#we shall pick the first 8 components for our prediction.

pca_new = PCA(n_components=8)

X_new = pca_new.fit_transform(X)
print(X_new)
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.25)
print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix, accuracy_score

lr = LogisticRegression()

lr.fit(X_train, y_train)

lr_predict = lr.predict(X_test)
# print confusion matrix and accuracy score

lr_confusion_matrix = confusion_matrix(y_test, lr_predict)

lr_accuracy_score = accuracy_score(y_test, lr_predict)

print(lr_confusion_matrix)

print(lr_accuracy_score*100)
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()

dt.fit(X_train,y_train)

dt_predict = dt.predict(X_test)
#print confusion matrix and accuracy score

dt_confusion_matrix = confusion_matrix(y_test, dt_predict)

dt_accuracy_score = accuracy_score(y_test, dt_predict)

print(dt_confusion_matrix)

print(dt_accuracy_score*100)
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(X_train, y_train)

nb_predict = nb.predict(X_test)
#print confusion matrix and accuracy score

nb_confusion_matrix = confusion_matrix(y_test, nb_predict)

nb_accuracy_score = accuracy_score(y_test, nb_predict)

print(nb_confusion_matrix)

print(nb_accuracy_score*100)
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()

rf.fit(X_train, y_train)

rf_predict = rf.predict(X_test)
# print confusion matrix and accuracy score

rf_confusion_matrix = confusion_matrix(y_test, rf_predict)

rf_accuracy_score = accuracy_score(y_test, rf_predict)

print(rf_confusion_matrix)

print(rf_accuracy_score*100)
from sklearn.svm import SVC

svc = SVC()

svc.fit(X_train, y_train)

svc_predict = svc.predict(X_test)
#print confusion matrix and accuracy score

svc_confusion_matrix = confusion_matrix(y_test, rf_predict)

svc_accuracy_score = accuracy_score(y_test, rf_predict)

print(svc_confusion_matrix)

print(svc_accuracy_score*100)
wine1 = [[7.8, 0.760, 0.04, 2.3, 0.092, 15.0, 54.0, 0.99700]]

print("Decision Tree : ",dt.predict(wine1))

print("Logistic Regression : ",lr.predict(wine1))

print("Naive Bayes : ",nb.predict(wine1))

print("Random forest : ",rf.predict(wine1))

print("SVM : ",svc.predict(wine1))