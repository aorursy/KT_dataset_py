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
df = pd.read_csv("/kaggle/input/greyscale_rice_leaf_disease_dataset.csv")
df.info()
df.describe()
df.head()
df.tail()
#Here the first column is the index column, the last column is the label



df_data = df.iloc[:, 1:]

df_data.info()
df_datadata.head()
X = df_data.iloc[:, 0:65536]

y = df_data.iloc[:, [65536]]
print(X.shape)

print(y.shape)
from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA

from sklearn.svm import SVC

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report

from sklearn.metrics import f1_score

from sklearn.metrics import recall_score

from sklearn.metrics import confusion_matrix
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, test_size = 0.2, random_state = 62)
#normalizing the data

X_train, X_test = X_train / 255.0, X_test / 255.0
from sklearn.preprocessing import StandardScaler
"""

sc = StandardScaler()

sc.fit(X_train)

X_train_std = sc.transform(X_train)

X_test_std = sc.transform(X_test)

"""
X_train_std = np.array(X_train_std)

X_test_std = np.array(X_test_std)

y_train = np.array(y_train)

y_test = np.array(y_test)
print(X_train_std.shape)

print(X_test_std.shape)

print(y_train.shape)

print(y_test.shape)
from sklearn.model_selection import GridSearchCV

parameters = [{'kernel': ['rbf'], 'C':[1, 10, 100, 1000], 'gamma':[.001, .0001, .00001]},

              {'kernel': ['linear'], 'C':[1, 10, 100, 1000]}]



clf = GridSearchCV(SVC(), param_grid = parameters, cv = 5, scoring='accuracy')

clf.fit(X_train, y_train.ravel())

print(clf.best_estimator_)

print(clf.best_params_)

print(clf.best_score_)
model = SVC(C=10, cache_size=200, class_weight=None, coef0=0.0,

    decision_function_shape='ovr', degree=3, gamma=1e-05, kernel='rbf',

    max_iter=-1, probability=False, random_state=None, shrinking=True,

    tol=0.001, verbose=True)

model.fit(X_train_std, y_train.ravel())
y_pred = model.predict(X_test_std)



acc = accuracy_score(y_test, y_pred)

print(acc)
#use X_train, X_test, y_train, y_test for any future task