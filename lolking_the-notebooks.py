# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn import preprocessing

from sklearn.metrics import accuracy_score

from sklearn.svm import LinearSVC

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Any results you write to the current directory are saved as output.

train = pd.read_csv('/kaggle/input/tdac-wine/Test_Data.csv')

print(train.shape)

train.head()
Val = pd.read_csv('/kaggle/input/tdac-wine/Val_Data.csv')

print(Val.shape)

Val.head()

# the objective is to predict type of of wine using logistic regression

X = train.drop(columns=['type','index'])

y = train.type

X.head()
# To see the bias of the data

ax = sns.barplot(x = [0,1],y = [(y[y ==0]).count(), (y[y ==1]).count()])

plt.show(ax)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

normalized_X = preprocessing.normalize(X)

normalized_X_train = preprocessing.normalize(X_train)

normalized_X_test = preprocessing.normalize(X_test)

normalized_X_val = preprocessing.normalize(Val)

clf = LinearSVC(random_state=0,C = 10)

clf.fit(normalized_X_train,y_train)

prediction = clf.predict(normalized_X_test)

accuracy_score(y_test, prediction, normalize=True)
from sklearn.model_selection import cross_val_score

clf = SVC(kernel='linear', C=1)

scores = cross_val_score(clf, X, y, cv=5)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
from sklearn import metrics

scores = cross_val_score(clf, X, y, cv=5, scoring='f1_macro')

print(scores)

clf.fit(X,y)

prediction = clf.predict(normalized_X_test)

accuracy_score(y_test, prediction, normalize=True)
clf = LinearSVC(random_state=0, C = 10)

clf.fit(normalized_X,y)

prediction = clf.predict(normalized_X_test)

accuracy_score(y_test, prediction, normalize=True)

prediction = clf.predict(Val.drop(columns = ['Index']))

output = pd.DataFrame({'ID': Val.Index,

                       'type': prediction})

output.to_csv('submission.csv', index=False)