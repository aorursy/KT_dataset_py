# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # visualization

import seaborn as sns #visualization



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
dataset = pd.read_csv('/kaggle/input/breast-cancer-csv/breastCancer.csv')
dataset.info()
dataset['bare_nucleoli'].unique()
dataset = dataset.drop('bare_nucleoli',axis=1)
dataset.describe()
dataset['class'].value_counts()
#dataset['class'].plot(kind='hist',color='green')



labels = '2-benign', '4-malignant'

sizes = dataset['class'].value_counts()

colors = ['green', 'red']

plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')

plt.show()
# correlation map

f,ax = plt.subplots(figsize=(10,10))

sns.heatmap(dataset.corr(), annot=True, linewidths=.5, fmt='.2f', ax=ax)

plt.show()
X = dataset.iloc[:, 1:-1].values

y = dataset.iloc[:, -1].values
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test) 
from sklearn.linear_model import LogisticRegression

classifier_lr = LogisticRegression(random_state = 0)

classifier_lr.fit(X_train, y_train)
from sklearn.metrics import confusion_matrix, accuracy_score

y_pred_lr = classifier_lr.predict(X_test)

cm_lr = confusion_matrix(y_test, y_pred_lr)

print(cm_lr)

accuracy_lr = accuracy_score(y_test, y_pred_lr)

print(accuracy_lr)

# accuracy list for algorithms

accuracy_list = []

accuracy_list.append(accuracy_lr)
from sklearn.neighbors import KNeighborsClassifier

classifier_knn = KNeighborsClassifier(n_neighbors=3, p=2, metric='minkowski') # n_neighbors = k

classifier_knn.fit(X_train, y_train)
from sklearn.metrics import confusion_matrix, accuracy_score

y_pred_knn = classifier_knn.predict(X_test)

cm_knn = confusion_matrix(y_test, y_pred_knn)

print(cm_knn)

accuracy_knn = accuracy_score(y_test, y_pred_knn)

print(accuracy_knn)

accuracy_list.append(accuracy_knn)
score_list = []

for each in range(1,15):

    classifier_knn_2 = KNeighborsClassifier(n_neighbors = each)

    classifier_knn_2.fit(X_train,y_train)

    score_list.append(classifier_knn_2.score(X_test,y_test))

    

plt.plot(range(1,15),score_list)

plt.xlabel("k values")

plt.ylabel("accuracy")

plt.show()
from sklearn.svm import SVC

classifier_svm = SVC(kernel = 'rbf', random_state = 0)

classifier_svm.fit(X_train, y_train)
from sklearn.metrics import confusion_matrix, accuracy_score

y_pred_svm = classifier_svm.predict(X_test)

cm_svm = confusion_matrix(y_test, y_pred_svm)

print(cm_svm)

accuracy_svm = accuracy_score(y_test, y_pred_svm)

print(accuracy_svm)

accuracy_list.append(accuracy_svm)
from sklearn.naive_bayes import GaussianNB

classifier_nb = GaussianNB()

classifier_nb.fit(X_train, y_train)
from sklearn.metrics import confusion_matrix, accuracy_score

y_pred_nb = classifier_nb.predict(X_test)

cm_nb = confusion_matrix(y_test, y_pred_nb)

print(cm_nb)

accuracy_nb = accuracy_score(y_test, y_pred_nb)

print(accuracy_nb)

accuracy_list.append(accuracy_nb)
from sklearn.tree import DecisionTreeClassifier

classifier_dt = DecisionTreeClassifier(criterion='gini', random_state=0)

classifier_dt.fit(X_train, y_train)
from sklearn.metrics import confusion_matrix, accuracy_score

y_pred_dt = classifier_dt.predict(X_test)

cm_dt = confusion_matrix(y_test, y_pred_dt)

print(cm_dt)

accuracy_dt = accuracy_score(y_test, y_pred_dt)

print(accuracy_dt)

accuracy_list.append(accuracy_dt)
from sklearn.ensemble import RandomForestClassifier

classifier_rf = RandomForestClassifier(n_estimators=10, criterion='gini', random_state = 0 )

classifier_rf.fit(X_train, y_train)
from sklearn.metrics import confusion_matrix,accuracy_score

y_pred_rf = classifier_rf.predict(X_test)

cm_rf = confusion_matrix(y_test, y_pred_rf)

print(cm_rf)

accuracy_rf = accuracy_score(y_test, y_pred_rf)

print(accuracy_rf)

accuracy_list.append(accuracy_rf)
models = ['lr', 'knn', 'svm', 'nb', 'dt', 'rf']

plt.plot(accuracy_list)

plt.scatter(models, accuracy_list)

plt.xlabel("Model")

plt.ylabel("Accuracy")
#%% Confusion Matrix visualization for Support Vector Machine

import seaborn as sns

f,ax = plt.subplots(figsize=(5,5))

sns.heatmap(cm_svm,annot=True,linewidths=0.5, linecolor="red",fmt=".0f",ax=ax)

plt.xlabel("y_pred_svm")

plt.ylabel("y_test")

plt.show