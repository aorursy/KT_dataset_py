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
from sklearn.datasets import load_iris
iris = load_iris()

iris.feature_names
iris.target_names
iris.data[0]
iris.target[0]
from sklearn.model_selection import train_test_split

train_x, test_x, train_y, test_y = train_test_split(iris.data, 

                                                    iris.target,

                                                   test_size=0.2, 

                                                   random_state=2)

from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier()

clf.fit(train_x, train_y)

y_hat = clf.predict(test_x)
from sklearn.metrics import accuracy_score

accuracy_score(test_y, y_hat)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()

knn.fit(train_x, train_y)

y_hat_knn = knn.predict(test_x)

accuracy_score(test_y, y_hat_knn)
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=10)

rf.fit(train_x, train_y)

y_hat_rf = rf.predict(test_x)

accuracy_score(test_y, y_hat_rf)
# visualization 

from sklearn.externals.six import StringIO

import pydot

from sklearn import tree

import matplotlib.image as mpimg





dot_data = StringIO()

tree.export_graphviz(clf, 

                    out_file=dot_data, 

                    feature_names=iris.feature_names,

                    class_names=iris.target_names,

                    filled=True, 

                    rounded=True, 

                    impurity=False)

graph=pydot.graph_from_dot_data(dot_data.getvalue())

graph.write_pdf("iris.pdf")

img = mpimg.imread(iris.pdf)

plt.figure(figsize=(100, 200))

plt.imshow(img,interpolation='nearest')
