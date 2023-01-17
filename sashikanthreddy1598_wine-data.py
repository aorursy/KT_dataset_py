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

import graphviz

import seaborn as sns

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_graphviz

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

import os

import missingno as msno
df = pd.read_csv("../input/winequality-red.csv")

df.head()
df.tail()
df.describe()
import matplotlib.pyplot as plt

# msno.bar(df)

# plt.show()


msno.bar(df)

plt.show()
df.info()
import graphviz
import matplotlib.pyplot as plt
# plt.rc('axis',lw = 1.5)

# plt.rc('xtick', labelsize = 14)

# plt.rc('ytick', labelsize = 14)

# plt.rc('xtick.major', size = 5, width = 3)

# plt.rc('ytick.major', size = 5, width = 3)
from sklearn.datasets import load_wine

wine = load_wine()

X= wine.data

y = wine.target
tree = DecisionTreeClassifier(max_depth = 2, random_state = 0)

tree.fit(X,y)
import graphviz
dot_data = export_graphviz(tree,

                out_file = None,

                feature_names = wine.feature_names,

                class_names=wine.target_names,

                rounded = True,

                filled = True)



graph = graphviz.Source(dot_data)

graph.render() 

graph
wine = load_wine()

X = wine.data[:,[6,12]] # flavanoids and proline

y = wine.target



# random_state is set to guarantee consistent result. You should remove it when running your own code.

tree1 = DecisionTreeClassifier(random_state=5) 

tree1.fit(X,y)
# preparing to plot the decision boundaries

x0min, x0max = X[:,0].min()-1, X[:,0].max()+1

x1min, x1max = X[:,1].min()-10, X[:,1].max()+10

xx0, xx1 = np.meshgrid(np.arange(x0min,x0max,0.02),np.arange(x1min, x1max,0.2))

Z = tree1.predict(np.c_[xx0.ravel(), xx1.ravel()])

Z = Z.reshape(xx0.shape)
plt.subplots(figsize=(12,10))

plt.contourf(xx0, xx1, Z, cmap=plt.cm.RdYlBu)

plot_colors = "ryb"

n_classes = 3

for i, color in zip(range(n_classes), plot_colors):

    idx = np.where(y == i)

    plt.scatter(X[idx, 0], X[idx, 1], c=color, label=wine.target_names[i],

                cmap=plt.cm.RdYlBu, edgecolor='black', s=30)

plt.legend(fontsize=18)

plt.xlabel('flavanoids', fontsize = 18)

plt.ylabel('proline', fontsize = 18)

plt.show()
# limit maximum tree depth

tree1 = DecisionTreeClassifier(max_depth=3,random_state=5) 

tree1.fit(X,y)



# limit maximum number of leaf nodes

tree2 = DecisionTreeClassifier(max_leaf_nodes=4,random_state=5) 

tree2.fit(X,y)



x0min, x0max = X[:,0].min()-1, X[:,0].max()+1

x1min, x1max = X[:,1].min()-10, X[:,1].max()+10

xx0, xx1 = np.meshgrid(np.arange(x0min,x0max,0.02),np.arange(x1min, x1max,0.2))



Z1 = tree1.predict(np.c_[xx0.ravel(), xx1.ravel()])

Z1 = Z1.reshape(xx0.shape)

Z2 = tree2.predict(np.c_[xx0.ravel(), xx1.ravel()])

Z2 = Z2.reshape(xx0.shape)



fig,ax = plt.subplots(nrows=1, ncols=2, figsize=(15,6))

ax[0].contourf(xx0, xx1, Z1, cmap=plt.cm.RdYlBu)

ax[1].contourf(xx0, xx1, Z2, cmap=plt.cm.RdYlBu)

plot_colors = "ryb"

n_classes = 3

for i, color in zip(range(n_classes), plot_colors):

    idx = np.where(y == i)

    ax[0].scatter(X[idx, 0], X[idx, 1], c=color, label=wine.target_names[i],

                cmap=plt.cm.RdYlBu, edgecolor='black', s=30)

    ax[1].scatter(X[idx, 0], X[idx, 1], c=color, label=wine.target_names[i],

                cmap=plt.cm.RdYlBu, edgecolor='black', s=30)

ax[0].legend(fontsize=14)

ax[0].set_xlabel('flavanoids', fontsize = 18)

ax[0].set_ylabel('proline', fontsize = 18)

ax[0].set_ylim(260,1690)

ax[0].set_title('max_depth = 3', fontsize = 14)

ax[1].legend(fontsize=14)

ax[1].set_xlabel('flavanoids', fontsize = 18)

ax[1].set_ylabel('proline', fontsize = 18)

ax[1].set_ylim(260,1690)

ax[1].set_title('max_leaf_nodes = 4', fontsize = 14)

plt.show()

