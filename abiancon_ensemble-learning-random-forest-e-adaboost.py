# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split

from sklearn import metrics

import seaborn as sns

import matplotlib.pyplot as plt
data = load_iris()

X = data.data

y = data.target



X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3)
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators = 50)

model = rfc.fit(X_train,y_train)

y_pred = model.predict(X_test)

cm = metrics.confusion_matrix(y_test, y_pred)

plt.title("Confusion Matrix")

sns.heatmap(cm,annot=True,fmt='.0f')

plt.show()

print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))
from sklearn.ensemble import AdaBoostClassifier

from sklearn.tree import DecisionTreeClassifier

adc = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators = 50, learning_rate = 1)

model1 = adc.fit(X_train, y_train)

y_pred = model1.predict(X_test)

cm = metrics.confusion_matrix(y_test, y_pred)

plt.title("Confusion Matrix")

sns.heatmap(cm,annot=True,fmt='.0f')

plt.show()

print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))
from sklearn.svm import SVC

svc=SVC(probability=True, kernel='linear')

abc =AdaBoostClassifier(n_estimators=50, base_estimator=svc,learning_rate=1)

model2 = abc.fit(X_train, y_train)

y_pred = model2.predict(X_test)

cm = metrics.confusion_matrix(y_test, y_pred)

plt.title("Confusion Matrix")

sns.heatmap(cm,annot=True,fmt='.0f')

plt.show()

print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))