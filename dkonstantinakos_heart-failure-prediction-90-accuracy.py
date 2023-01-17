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
import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
dataset = pd.read_csv('/kaggle/input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv')
dataset.info()
dataset.head()
plt.figure(figsize = (12,6))

sns.countplot(x = 'smoking', hue = 'DEATH_EVENT', data = dataset)
dataset.groupby(['smoking', 'DEATH_EVENT']).count()
plt.figure(figsize = (12,6))

sns.countplot(x = 'high_blood_pressure', hue = 'DEATH_EVENT', data = dataset)
dataset.groupby(['high_blood_pressure', 'DEATH_EVENT']).count()
plt.figure(figsize = (12,6))

sns.countplot(x = 'anaemia', hue = 'DEATH_EVENT', data = dataset)
dataset.groupby(['anaemia', 'DEATH_EVENT']).count()
plt.figure(figsize=(12,6))

sns.countplot(x = 'diabetes', hue = 'DEATH_EVENT', data = dataset)
dataset.groupby(['diabetes', 'DEATH_EVENT']).count()
sns.boxplot(x = 'DEATH_EVENT', y = 'creatinine_phosphokinase', data = dataset)
sns.boxplot(x = 'DEATH_EVENT', y = 'ejection_fraction', data = dataset)
sns.boxplot(x = 'DEATH_EVENT', y = 'platelets', data = dataset)
sns.boxplot(x = 'DEATH_EVENT', y = 'serum_creatinine', data = dataset)
sns.boxplot(x = 'DEATH_EVENT', y = 'serum_sodium', data = dataset)
sns.boxplot(x = 'DEATH_EVENT', y = 'time', data = dataset)
sns.boxplot(x = 'DEATH_EVENT', y = 'age', data = dataset)
plt.figure(figsize = (16,10))

corr = dataset.corr()

mask = np.zeros_like(corr)

mask[np.triu_indices_from(mask)] = True

sns.heatmap(corr, mask = mask, annot = True, cmap = 'viridis')
corr['DEATH_EVENT'].drop('DEATH_EVENT').sort_values(ascending=True).plot.bar()
X = dataset.loc[:, ['serum_creatinine','ejection_fraction', 'time']].values

y = dataset.iloc[:, -1].values

# Splitting to training set and test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

# Scaling the data

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
from sklearn.svm import SVC

classifier = SVC(kernel = 'linear', random_state = 0)

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score, recall_score

cm = confusion_matrix(y_test, y_pred)

print(cm)

print("Accuracy: {:.2f} %".format(accuracy_score(y_test, y_pred)*100))

print("Recall: {:.2f} %".format(recall_score(y_test, y_pred)*100))
from sklearn.svm import SVC

classifier = SVC(kernel = 'rbf', random_state = 0)

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score, recall_score

cm = confusion_matrix(y_test, y_pred)

print(cm)

print("Accuracy: {:.2f} %".format(accuracy_score(y_test, y_pred)*100))

print("Recall: {:.2f} %".format(recall_score(y_test, y_pred)*100))
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state = 0)

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score, recall_score

cm = confusion_matrix(y_test, y_pred)

print(cm)

print("Accuracy: {:.2f} %".format(accuracy_score(y_test, y_pred)*100))

print("Recall: {:.2f} %".format(recall_score(y_test, y_pred)*100))
from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score, recall_score

cm = confusion_matrix(y_test, y_pred)

print(cm)

print("Accuracy: {:.2f} %".format(accuracy_score(y_test, y_pred)*100))

print("Recall: {:.2f} %".format(recall_score(y_test, y_pred)*100))
from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score, recall_score

cm = confusion_matrix(y_test, y_pred)

print(cm)

print("Accuracy: {:.2f} %".format(accuracy_score(y_test, y_pred)*100))

print("Recall: {:.2f} %".format(recall_score(y_test, y_pred)*100))
from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score, recall_score

cm = confusion_matrix(y_test, y_pred)

print(cm)

print("Accuracy: {:.2f} %".format(accuracy_score(y_test, y_pred)*100))

print("Recall: {:.2f} %".format(recall_score(y_test, y_pred)*100))
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators = 50, criterion = 'entropy', random_state = 0)

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score, recall_score

cm = confusion_matrix(y_test, y_pred)

print(cm)

print("Accuracy: {:.2f} %".format(accuracy_score(y_test, y_pred)*100))

print("Recall: {:.2f} %".format(recall_score(y_test, y_pred)*100))
from catboost import CatBoostClassifier

classifier = CatBoostClassifier()

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score, recall_score

cm = confusion_matrix(y_test, y_pred)

print(cm)

print("Accuracy: {:.2f} %".format(accuracy_score(y_test, y_pred)*100))

print("Recall: {:.2f} %".format(recall_score(y_test, y_pred)*100))
from xgboost import XGBClassifier

classifier = XGBClassifier()

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score, recall_score

cm = confusion_matrix(y_test, y_pred)

print(cm)

print("Accuracy: {:.2f} %".format(accuracy_score(y_test, y_pred)*100))

print("Recall: {:.2f} %".format(recall_score(y_test, y_pred)*100))
from sklearn.ensemble import GradientBoostingClassifier

classifier = GradientBoostingClassifier(max_depth=2, random_state=4)

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score, recall_score

cm = confusion_matrix(y_test, y_pred)

print(cm)

print("Accuracy: {:.2f} %".format(accuracy_score(y_test, y_pred)*100))

print("Recall: {:.2f} %".format(recall_score(y_test, y_pred)*100))
from catboost import CatBoostClassifier

classifier = CatBoostClassifier()

classifier.fit(X_train, y_train)
from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)

print("Accuracy: {:.2f} %".format(accuracies.mean()*100))

print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))
from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 5)

print("Accuracy: {:.2f} %".format(accuracies.mean()*100))

print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))