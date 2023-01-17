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
import numpy as np

import pandas as pd

import seaborn as sns

sns.set_palette('husl')

import matplotlib.pyplot as plt

%matplotlib inline



from sklearn import metrics

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split


Iris = pd.read_csv("../input/Iris.csv")
Iris.head()
Iris.info()
Iris.describe()
Iris['Species'].value_counts()
tmp=Iris.drop('Id',axis=1)

g=sns.pairplot(tmp,hue='Species',markers='+')

plt.show()
g = sns.violinplot(y='Species', x='SepalLengthCm', data=Iris, inner='quartile')

plt.show()

g = sns.violinplot(y='Species', x='SepalWidthCm', data=Iris, inner='quartile')

plt.show()

g = sns.violinplot(y='Species', x='PetalLengthCm', data=Iris, inner='quartile')

plt.show()

g = sns.violinplot(y='Species', x='PetalWidthCm', data=Iris, inner='quartile')

plt.show()
X = Iris.drop(['Id', 'Species'], axis=1)

y = Iris['Species']

print(X.head())

print(X.shape)

print(y.head())

print(y.shape)
# experimenting with different n values

k_range = list(range(1,26))

scores = []

for k in k_range:

    knn = KNeighborsClassifier(n_neighbors=k)

    knn.fit(X, y)

    y_pred = knn.predict(X)

    scores.append(metrics.accuracy_score(y, y_pred))

    

plt.plot(k_range, scores)

plt.xlabel('Value of k for KNN')

plt.ylabel('Accuracy Score')

plt.title('Accuracy Scores for Values of k of k-Nearest-Neighbors')

plt.show()
logreg = LogisticRegression()

logreg.fit(X, y)

y_pred = logreg.predict(X)

print(metrics.accuracy_score(y, y_pred))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=5)

print(X_train.shape)

print(y_train.shape)

print(X_test.shape)

print(y_test.shape)
# experimenting with different n values

k_range = list(range(1,26))

scores = []

for k in k_range:

    knn = KNeighborsClassifier(n_neighbors=k)

    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)

    scores.append(metrics.accuracy_score(y_test, y_pred))

    

plt.plot(k_range, scores)

plt.xlabel('Value of k for KNN')

plt.ylabel('Accuracy Score')

plt.title('Accuracy Scores for Values of k of k-Nearest-Neighbors')

plt.show()
logreg = LogisticRegression()

logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)

print(metrics.accuracy_score(y_test, y_pred))
knn = KNeighborsClassifier(n_neighbors=12)

knn.fit(X, y)



# make a prediction for an example of an out-of-sample observation

knn.predict([[6, 3, 4, 2]])