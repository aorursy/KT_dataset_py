import pandas as pd

wine = pd.read_csv("../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv")
wine.head()
wine.describe()
# Data visualization

import seaborn as sns

import matplotlib.pyplot as plt

sns.countplot(x='quality',data=wine)
# Cleaning Data

wine['fixed acidity'].fillna(wine['fixed acidity'].median(), inplace = True)

wine['volatile acidity'].fillna(wine['volatile acidity'].median(), inplace = True)

wine['citric acid'].fillna(wine['citric acid'].median(), inplace = True)

wine['residual sugar'].fillna(wine['residual sugar'].median(), inplace = True)

wine['chlorides'].fillna(wine['chlorides'].median(), inplace = True)

wine['free sulfur dioxide'].fillna(wine['free sulfur dioxide'].median(), inplace = True)

wine['total sulfur dioxide'].fillna(wine['total sulfur dioxide'].median(), inplace = True)

wine['density'].fillna(wine['density'].median(), inplace = True)

wine['pH'].fillna(wine['pH'].median(), inplace = True)

wine['sulphates'].fillna(wine['sulphates'].median(), inplace = True)

wine['alcohol'].fillna(wine['alcohol'].median(), inplace = True)
# Rating

rating = []

for i in wine['quality']:

    if i >= 1 and i <= 3:

        rating.append('1')

    elif i >= 4 and i <= 7:

        rating.append('2')

    elif i >= 8 and i <= 10:

        rating.append('3')

wine['Rating'] = rating
wine.head()
X = wine.iloc[:,:11]

y = wine['Rating']
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X = sc.fit_transform(X)
# PCA

from sklearn.decomposition import PCA

pca = PCA()

X_pca = pca.fit_transform(X)

pca_new = PCA(n_components=8)

X_new = pca_new.fit_transform(X)
# Splitting Data

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.25)
# Naive Bayes

from sklearn.naive_bayes import GaussianNB

from sklearn import metrics

gnb = GaussianNB()

gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)

print("Accuracy: ",metrics.accuracy_score(y_test, y_pred))
# Confusion Matrix

from sklearn.metrics import confusion_matrix

confusion_matrix(y_test,y_pred)

pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)
# Classification Report

from sklearn.metrics import classification_report

print(classification_report(y_test,y_pred))