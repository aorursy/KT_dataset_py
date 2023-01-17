import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

iris = pd.read_csv('../input/Iris.csv')
iris.head()
iris.describe()
iris.Species.describe()
iris.shape
irisdata = iris.drop(['Id'], axis = 1)
irisdata.plot(kind='hist', subplots=True, layout =(2,2))
irisdata.plot(kind='box', subplots=True, layout =(2,2))
import sklearn.model_selection as ms
X = iris.iloc[:, 1:5]

y = iris.loc[:, 'Species']
X_train, X_test, y_train, y_test = ms.train_test_split(X, y, test_size=0.33, random_state=42)
from sklearn import tree

from sklearn.tree import export_graphviz

from subprocess import check_call
clf = tree.DecisionTreeClassifier()

treemodel = clf.fit(X_train, y_train)
from IPython.display import Image as Img
with open("treeop.dot", 'w') as f:

     treeop = tree.export_graphviz(treemodel, out_file='treeop.dot', feature_names = X.columns, filled = True)



check_call(['dot','-Tpng','treeop.dot','-o','tree1.png'])

Img("tree1.png")