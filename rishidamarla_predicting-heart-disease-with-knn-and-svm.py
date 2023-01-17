import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

from sklearn.metrics import f1_score

from sklearn.model_selection import train_test_split

import warnings

warnings.filterwarnings('ignore')

import sklearn.neighbors

from sklearn.neighbors import KNeighborsClassifier
df = pd.read_csv('../input/heart-disease-prediction/Heart_Disease_Prediction.csv')
df.head()
df.describe()
df.shape
# Checking for missing values.

df.isnull().values.any()
# Checking for imbalanced data based on sex.

df['Sex'].value_counts()
# Checking for imbalanced data based on outcome.

df['Heart Disease'].value_counts()
sns.countplot(x='Heart Disease', data=df)
# Checking for any correlations.

df.corr()
# Splitting the dataset into training and testing sets.

x = df.iloc[:, :-2]

y = df.iloc[:, -1]

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 0, test_size = 0.35)
# Using standard scaler as a standardization technique.

sc_x = StandardScaler()

x_train = sc_x.fit_transform(x_train)

x_test = sc_x.transform(x_test)
# Looking for optimal number of nearest neighbours.

import math

math.sqrt(len(y_test))
# Creating KNN Model.

classifier = KNeighborsClassifier(n_neighbors = 9, p = 2, metric = 'euclidean')

classifier.fit(x_train,y_train)
y_pred = classifier.predict(x_test)

y_pred
cm = confusion_matrix(y_test,y_pred)

print(cm)
print(accuracy_score(y_test,y_pred))
# Creating SVM model.

from sklearn import svm

clf = svm.SVC(kernel='rbf')

clf.fit(x_train,y_train)

y_pred = clf.predict(x_test)
y_pred = clf.predict(x_test)

y_pred
cm = confusion_matrix(y_test,y_pred)

print(cm)
print(accuracy_score(y_test,y_pred))