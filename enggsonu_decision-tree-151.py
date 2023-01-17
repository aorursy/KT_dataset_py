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

import seaborn as sn

import numpy as np

import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer,load_iris

from sklearn.model_selection import train_test_split
iris = load_iris()

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data,cancer.target,stratify=cancer.target,random_state=0)
x,y=cancer['data'],cancer['target']
cancer.feature_names
df=pd.DataFrame(x,columns=cancer.feature_names)
df.head()
df.tail()
df.info()
df.shape
df['target']=cancer['target']
df.head()
df_x=df[df.columns[:30]]
from sklearn.tree import DecisionTreeClassifier, export_graphviz

tree = DecisionTreeClassifier(max_depth=2)

tree.fit(X_train, y_train)
print("Feature importances:\n{}".format(tree.feature_importances_))
def plot_feature_importances_cancer(model):

    n_features = cancer.data.shape[1]

    plt.rcParams["figure.figsize"] = (10,10)

    plt.barh(range(n_features), model.feature_importances_, align='center')

    plt.yticks(np.arange(n_features), cancer.feature_names)

    plt.xlabel("Feature importance")

    plt.ylabel("Feature")
plot_feature_importances_cancer(tree)
from sklearn.tree import plot_tree

tree_dot = plot_tree(tree,feature_names=cancer.feature_names)
tree = DecisionTreeClassifier(max_depth=4)

tree.fit(X_train,y_train)
tree_dot = plot_tree(tree,feature_names=cancer.feature_names) 
from sklearn.model_selection import GridSearchCV

param_grid = {"max_depth":range(1,7)}

grid=GridSearchCV(DecisionTreeClassifier(random_state=0),

param_grid = param_grid,cv=10)

grid.fit(X_train,y_train)
print("Best parameters set found on developmentset:")

print(grid.best_params_)
print(grid.score(X_test,y_test))
print("Best estimator:\n{}".format(grid.best_estimator_))
results=pd.DataFrame(grid.cv_results_)

display(results)
param_grid = {'max_leaf_nodes':range(2, 20)}

grid = GridSearchCV(DecisionTreeClassifier(random_state=0),

param_grid=param_grid, cv=10)

grid.fit(X_train, y_train)
X_train, X_test, y_train, y_test = train_test_split(

iris.data, iris.target, stratify=iris.target, random_state=0)

tree = DecisionTreeClassifier(max_leaf_nodes=6)

tree.fit(X_train, y_train)

tree_dot = plot_tree(tree, feature_names=cancer.feature_names)
X_train, X_test, y_train, y_test = train_test_split(

iris.data, iris.target, stratify=iris.target, random_state=2)

tree = DecisionTreeClassifier(max_leaf_nodes=6)

tree.fit(X_train, y_train)

tree_dot = plot_tree(tree, feature_names=cancer.feature_names)