# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
filename = "../input/winequality-red.csv"

data = pd.read_csv(filename, delimiter=',', encoding='utf-8')
data.head()
data.describe()
data['label'] = data['quality'] > 6
data['label'].value_counts()
data.head()
X = data.iloc[:,:len(data.columns) - 2]

X.head()
y = data.iloc[:,len(data.columns) - 1]

y.head()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 42)
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import zero_one_loss

from sklearn.ensemble import AdaBoostClassifier
#dtree = DecisionTreeClassifier(max_depth = 4, min_samples_leaf = 5)

dtree = DecisionTreeClassifier()

dtree.fit(X_train,y_train)
import graphviz

from graphviz import Source

from sklearn import tree

Source(tree.export_graphviz(dtree, out_file=None, feature_names=X.columns))
dtree_error = 1.0 - dtree.score(X_test, y_test)

dtree_error
from sklearn.model_selection import GridSearchCV

parameters = {'max_depth': np.arange(1,10), 'min_samples_leaf': np.arange(1,10)}
treeGrid = GridSearchCV(dtree, parameters, cv=5)

treeGrid.fit(X_train, y_train)
treeGrid_error = 1.0 - treeGrid.score(X_test, y_test)

treeGrid_error
from sklearn.metrics import average_precision_score

average_precision_score(y_test, treeGrid.predict(X_test))
import pickle

with open('../filename.pickle', 'wb') as handle:

    pickle.dump(treeGrid, handle, protocol=pickle.HIGHEST_PROTOCOL)