import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



import warnings

warnings.filterwarnings('ignore')
df = pd.read_csv('../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')
df.head()
df.shape
df.columns
df.info()
df.describe()
df.isnull().sum()
df['quality'].unique()
df.quality.value_counts().sort_index()
sns.countplot(x='quality', data = df)
reviews = []

for i in df['quality']:

    if i >= 1 and i <= 4:

        reviews.append('1')

    elif i >= 5 and i <= 6:

        reviews.append('2')

    elif i >= 7 and i <= 8:

        reviews.append('3')

df['Reviews'] = reviews
df.Reviews.value_counts()

# poor = 1

# average = 2

# good = 3
corrmat = df.corr()

plt.figure(figsize=(20,10))

sns.heatmap(corrmat, annot=True, cmap='coolwarm')
corrmat['quality'].sort_values(ascending = False)
sns.boxplot('quality', 'alcohol', data = df)
sns.boxplot('Reviews', 'sulphates', data = df)
sns.boxplot('Reviews', 'citric acid', data = df)
sns.boxplot('Reviews', 'fixed acidity', data = df)
sns.boxplot('Reviews', 'residual sugar', data = df)
X = df.iloc[:,0:11]

y = df['Reviews']
X.head(10)
y.head(10)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)
print(X_train.shape)

print(X_test.shape)

#print(y_train.shape)

#print(y_test.shape)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
l=[]
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

model = LogisticRegression()

model.fit(X_train, y_train)



y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)

print('Logistic Regression:', acc * 100)

l.append(acc)
from sklearn.svm import SVC

classifier = SVC(kernel = 'rbf', random_state = 42)

classifier.fit(X_train, y_train)



y_pred = classifier.predict(X_test)

acc = accuracy_score(y_test, y_pred)

print('SVM:', acc * 100)

l.append(acc)
from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)

classifier.fit(X_train, y_train)



y_pred = classifier.predict(X_test)

acc = accuracy_score(y_test, y_pred)

print('Decision Tree:', acc * 100)

l.append(acc)
from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()

classifier.fit(X_train, y_train)



y_pred = classifier.predict(X_test)

acc = accuracy_score(y_test, y_pred)

print('Naive Bayes:', acc * 100)

l.append(acc)
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators = 30, criterion = 'entropy', random_state = 0)

classifier.fit(X_train, y_train)



y_pred = classifier.predict(X_test)

acc = accuracy_score(y_test, y_pred)

print('Random Forest:',acc * 100)

l.append(acc)
from sklearn.metrics import classification_report as cr

print(cr(y_test, y_pred))
from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors = 6, metric = 'minkowski', p = 2)

classifier.fit(X_train, y_train)



y_pred = classifier.predict(X_test)

acc = accuracy_score(y_test, y_pred)

print('Knn:',acc * 100)

l.append(acc)
y_axis=['Logistic Regression',

     'Support Vector Classifier',

      'Decision Tree Classifier',

       'Gaussian Naive Bayes',

      'Random Forest Classifier',

       'K-Neighbors Classifier']

x_axis=l

sns.barplot(x=x_axis,y=y_axis)

plt.xlabel('Accuracy')