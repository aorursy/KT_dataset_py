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
df = pd.read_csv('/kaggle/input/iris/Iris.csv')

df.head()
df.isna().sum()
df.drop('Id', axis=1, inplace=True)
import seaborn as sns 

import matplotlib.pyplot as plt

plt.figure(figsize=(10,10))

plt.subplot(2,2,1)

sns.barplot(x = df['Species'], y = df['SepalLengthCm'], data=df)

plt.subplot(2,2,2)

sns.barplot(x = df['Species'], y = df['SepalWidthCm'], data=df)

plt.subplot(2,2,3)

sns.barplot(x = df['Species'], y = df['PetalLengthCm'], data=df)

plt.subplot(2,2,4)

sns.barplot(x = df['Species'], y = df['PetalWidthCm'], data=df)
sns.pairplot(df, hue="Species")
df.plot(kind='area', figsize=(10,10), alpha=0.75)
X = df.iloc[:, :-1].values

Y = df.iloc[:, -1].values
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 1)
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