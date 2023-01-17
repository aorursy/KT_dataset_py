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
hr_df = pd.read_csv('../input/hr-analytics/HR_comma_sep.csv')

hr_df.head()
hr_df.shape
hr_df = hr_df.dropna()

hr_df.shape
from sklearn.model_selection import train_test_split
y = hr_df['salary']

y
prediktor = ['satisfaction_level', 'number_project', 'time_spend_company', 'average_montly_hours']

X = hr_df[prediktor]

X
X.describe()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 10)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
k_range = list(range(1,26))

nilai = []

for k in k_range:

    model_knn = KNeighborsClassifier(n_neighbors = k)

    model_knn.fit(X_train, y_train)

    y_pred = model_knn.predict(X_test)

    nilai.append(accuracy_score(y_test, y_pred))
model_knn = KNeighborsClassifier(n_neighbors = 3)

model_knn.fit(X_train, y_train)

y_pred = model_knn.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
from sklearn.linear_model import LogisticRegression
model_logreg = LogisticRegression(solver = 'lbfgs', multi_class = 'auto')

model_logreg.fit(X_train, y_train)

y_pred = model_logreg.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
from sklearn.svm import SVC
model_svc = SVC(gamma = 'scale')

model_svc.fit(X_train, y_train)

y_pred = model_svc.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
from sklearn.tree import DecisionTreeClassifier
model_dt = DecisionTreeClassifier()

model_dt.fit(X_train, y_train)

y_pred = model_dt.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
from sklearn.ensemble import RandomForestClassifier
model_dt = DecisionTreeClassifier()

model_dt.fit(X_train, y_train)

y_pred = model_dt.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))