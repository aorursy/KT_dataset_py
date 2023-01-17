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

test = pd.read_csv("../input/data-science-london-scikit-learn/test.csv", header=None)

train = pd.read_csv("../input/data-science-london-scikit-learn/train.csv", header=None)

trainLabels = pd.read_csv("../input/data-science-london-scikit-learn/trainLabels.csv", header=None)
print(train.shape)

print(test.shape)

print(trainLabels.shape)
train.head(10)
train.describe()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train, trainLabels, train_size = 0.3, random_state=0)
from sklearn.linear_model import LogisticRegression

lgrs = LogisticRegression(solver='lbfgs', penalty='none')

lgrs.fit(X_train, y_train)

lgrs_pred = lgrs.predict(X_test)

print("Train score: ", lgrs.score(X_train, y_train))

print("Test score: ", lgrs.score(X_test, y_test))
from sklearn.tree import DecisionTreeClassifier

dct = DecisionTreeClassifier(max_depth=5,random_state=0)

dct.fit(X_train,y_train)

print("Train score: ", dct.score(X_train, y_train))

print("Test score: ", dct.score(X_test, y_test))
dct_test = dct.predict(test)

dct_test.shape
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score, train_test_split

X, y = train, np.ravel(trainLabels)



X_train, X_test, y_train, y_test = train_test_split(train, trainLabels, test_size = 0.3, random_state=0)

neig = np.arange(1,25)

kfold = 10 

train_accuracy = []

val_accuracy = []

bestknn = None

bestacc = 0.0

for i, k in enumerate(neig):

    knn = KNeighborsClassifier(n_neighbors=k)

    knn.fit(X_train, y_train)

    train_accuracy.append(knn.score(X_train, y_train))

    val_accuracy.append(np.mean(cross_val_score(knn, X,y,cv=kfold)))

    if np.mean(cross_val_score(knn, X, y, cv=kfold)) > bestacc:

        bestacc = np.mean(cross_val_score(knn, X, y, cv=10))

        bestknn = knn

print("Best Accuracy :", bestacc)

print(bestknn)
knn_test = bestknn.predict(test)

knn_test.shape
submission = pd.DataFrame(knn_test)

print(submission.shape)

submission.columns = ['Solution']

submission['ID'] = np.arange(1,submission.shape[0]+1)

submission = submission[['ID', 'Solution']]

submission
filename = 'Final Test.csv'

submission.to_csv (filename, index=False)
