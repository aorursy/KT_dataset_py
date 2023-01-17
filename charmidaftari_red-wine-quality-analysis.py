# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')

df.head()
df.isna().sum()
df.describe()
import matplotlib.pyplot as plt

import seaborn as sns



fig = plt.figure(figsize = (10,6))

sns.barplot(x = 'quality', y='fixed acidity', data=df)

plt.title('quality vs fixed acidity')
# volatile acidity descreses as quality increases

fig = plt.figure(figsize = (10,6))

sns.barplot(x = 'quality', y='volatile acidity', data=df)

plt.title('quality vs volatile acidity')
# citric acid goes higher as quality increases

fig = plt.figure(figsize = (10,6))

sns.barplot(x = 'quality', y= 'citric acid', data=df)

plt.title('quality vs citric acid')
fig = plt.figure(figsize = (10,6))

sns.barplot(x = 'quality', y='residual sugar', data=df)

plt.title('quality vs residual sugar')
# level of chlorides decreases as quality increases

fig = plt.figure(figsize = (10,6))

sns.barplot(x = 'quality', y='chlorides', data=df)

plt.title('quality vs chlorides')
fig = plt.figure(figsize = (10,6))

sns.barplot(x = 'quality', y='free sulfur dioxide', data=df)

plt.title('quality vs free sulfur dioxide')
fig = plt.figure(figsize = (10,6))

sns.barplot(x = 'quality', y='total sulfur dioxide', data=df)

plt.title('quality vs total sulfur dioxide')
# alcohol level increases as quality increases

fig = plt.figure(figsize = (10,6))

sns.barplot(x = 'quality', y='alcohol', data=df)

plt.title('quality vs alcohol')
# Dividing the quality variable into good and bad classification

bins = (2, 6.5, 8)

labels = ['bad', 'good']

df['quality'] = pd.cut(df['quality'], bins = bins, labels = labels)
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

df['quality'] = le.fit_transform(df['quality'])
df['quality'].value_counts()
df['quality'].value_counts().plot(kind = 'pie', figsize = (5,6), autopct = '%1.1f%%', shadow = True, explode = [0.1, 0])
sns.countplot(df['quality'])
X = df.iloc[:, :-1].values

Y = df.iloc[:, -1].values
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 1)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.fit_transform(X_test)
print(X_train)
print(Y_train)
from sklearn.linear_model import LogisticRegression

classifier_logistic = LogisticRegression(random_state = 0)

classifier_logistic.fit(X_train, Y_train)
from sklearn.metrics import confusion_matrix, accuracy_score

pred_logistic = classifier_logistic.predict(X_test)

cm = confusion_matrix(Y_test, pred_logistic)

print(cm)

accuracy_score(Y_test, pred_logistic)
from sklearn.svm import SVC

classifier_svc = SVC()

classifier_svc.fit(X_train, Y_train)
from sklearn.metrics import confusion_matrix, accuracy_score

pred_svc = classifier_svc.predict(X_test)

cm = confusion_matrix(Y_test, pred_svc)

print(cm)

accuracy_score(Y_test, pred_svc)
from sklearn.neighbors import KNeighborsClassifier

classifier_kneighbors = KNeighborsClassifier()

classifier_kneighbors.fit(X_train, Y_train)
from sklearn.metrics import confusion_matrix, accuracy_score

pred_kneighbors = classifier_kneighbors.predict(X_test)

cm = confusion_matrix(Y_test, pred_kneighbors)

print(cm)

accuracy_score(Y_test, pred_kneighbors)
from sklearn.naive_bayes import GaussianNB

classifier_naive = GaussianNB()

classifier_naive.fit(X_train, Y_train)
from sklearn.metrics import confusion_matrix, accuracy_score

pred_naive = classifier_naive.predict(X_test)

cm = confusion_matrix(Y_test, pred_naive)

print(cm)

accuracy_score(Y_test, pred_naive)
from sklearn.tree import DecisionTreeClassifier

classifier_tree = DecisionTreeClassifier()

classifier_tree.fit(X_train, Y_train)
from sklearn.metrics import confusion_matrix, accuracy_score

pred_tree = classifier_tree.predict(X_test)

cm = confusion_matrix(Y_test, pred_tree)

print(cm)

accuracy_score(Y_test, pred_tree)
from sklearn.ensemble import RandomForestClassifier

classifier_forest = RandomForestClassifier()

classifier_forest.fit(X_train, Y_train)
from sklearn.metrics import confusion_matrix, accuracy_score

pred_forest = classifier_forest.predict(X_test)

cm = confusion_matrix(Y_test, pred_forest)

print(cm)

accuracy_score(Y_test, pred_forest)
from sklearn.model_selection import cross_val_score

eval_forest = cross_val_score(estimator = classifier_forest, X = X_train, y = Y_train, cv= 10)

eval_forest.mean()