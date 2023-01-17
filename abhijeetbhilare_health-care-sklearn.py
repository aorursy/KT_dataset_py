import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import plot_confusion_matrix

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')



csv_filename = ""

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        csv_filename = os.path.join(dirname, filename)

        print(os.path.join(dirname, filename))
df = pd.read_csv(csv_filename)

y = df.target

x = df.drop(columns=["target"])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
sns.pairplot(df[["age", "sex", "chol", "fbs", "target"]])
clf = LogisticRegression(random_state=0).fit(x_train, y_train)

pred = clf.predict(x_test)

print(accuracy_score(y_test, pred))

plot_confusion_matrix(clf, x_test, y_test)
from sklearn.ensemble import RandomForestClassifier

ranfor = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0,

                                   max_features = 'auto', max_depth = 10)

ranfor.fit(x_train, y_train)

pred_ranfor = ranfor.predict(x_test)

print(accuracy_score(y_test, pred_ranfor))

plot_confusion_matrix(ranfor, x_test, y_test)
from sklearn import svm

sv = svm.SVC(kernel='linear')

sv.fit(x_train, y_train)

pred_svm = sv.predict(x_test)

print(accuracy_score(y_test, pred_svm))

plot_confusion_matrix(sv, x_test, y_test)