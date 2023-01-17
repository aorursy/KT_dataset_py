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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import statsmodels.formula.api as sm

import pandas as pd

dataset = pd.read_csv("../input/haberman.csv")

X = dataset.iloc[:, [0,1,2]].values

y = dataset.iloc[:, -1].values
sns.swarmplot(y, X[:, 0])

plt.show()
sns.regplot(X[:, 0], y, data = dataset)

plt.show()
sns.swarmplot(y, X[:, 1])

plt.show()
sns.regplot(X[:, 1], y, data = dataset)

plt.show()
sns.swarmplot(y, X[:, 2])

plt.show()
sns.regplot(X[:, 2], y, data = dataset)

plt.show()
sns.pairplot(data = dataset )

plt.show()
X = dataset.iloc[:, 2:3].values
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
from sklearn.svm import SVC

classifier = SVC(kernel = 'rbf', random_state = 0)

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
y_pred
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
cm
accuracy = (51)/(77)*100
accuracy
from mlxtend.plotting import plot_decision_regions

plot_decision_regions(X_train, y_train ,clf = classifier)
from sklearn.svm import SVC

classifier = SVC(kernel = 'sigmoid', random_state = 0)

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
y_pred
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
cm
accuracy = (47+3)/(77)*100
accuracy
from mlxtend.plotting import plot_decision_regions

plot_decision_regions(X_train, y_train ,clf = classifier)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators = 1000, criterion = 'entropy', random_state = 0)

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
cm
accuracy = (47)/77*100
accuracy
from mlxtend.plotting import plot_decision_regions

plot_decision_regions(X_train, y_train ,clf = classifier)