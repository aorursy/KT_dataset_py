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
dataset = pd.read_csv('../input/iris/Iris.csv', index_col='Id')

dataset.head()
# Check whether dataset contains empty values

print(sum(dataset.isnull().values))
X_full = dataset.iloc[:, :-1]

y_full = dataset.iloc[:, -1:]



print(X_full.shape, y_full.shape)

print(X_full[:5])
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.3)
# from sklearn.preprocessing import StandardScaler

# ss = StandardScaler()

# X_train = ss.fit_transform(X_train)

# X_test = ss.transform(X_test)
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import GridSearchCV, cross_val_score



clf = KNeighborsClassifier()

param_grid = {'n_neighbors':[2, 3, 4, 5, 6, 7, 8, 9, 10],

              'leaf_size':[10, 15, 20, 25, 30, 35, 40]}

grid_search = GridSearchCV(clf, param_grid, cv=10, scoring='accuracy')

grid_search.fit(X_train, y_train)

y_pred = grid_search.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_test, y_pred)

acc = accuracy_score(y_test, y_pred)



print(acc)



import seaborn as sns

sns.heatmap(cm, annot=True)