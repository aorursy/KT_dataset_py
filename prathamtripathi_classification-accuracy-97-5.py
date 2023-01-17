import numpy as np

import pandas as pd

from sklearn import tree

import matplotlib.pyplot as plt

from matplotlib.ticker import NullFormatter

import pylab as py

import matplotlib.ticker as ticker

from sklearn import preprocessing

import collections

import matplotlib.image as mpimg

%matplotlib inline
!pip install pydotplus
import pydotplus
my_data=pd.read_csv('../input/drug-classification/drug200.csv',delimiter=',')

my_data[0:5]
X=my_data[['Age', 'Sex', 'BP', 'Cholesterol','Na_to_K']].values

X[0:5]
from sklearn.preprocessing import LabelEncoder

le_sex = LabelEncoder()

le_sex.fit(["M","F"])

X[:,1] = le_sex.transform(X[:,1])
le_BP = LabelEncoder()

le_BP.fit(["LOW","NORMAL","HIGH"])

X[:,2] = le_BP.transform(X[:,2])
le_chol = LabelEncoder()

le_chol.fit(["HIGH","NORMAL"])

X[:,3] = le_chol.transform(X[:,3])
y = my_data["Drug"].values

y[0:5]
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.4, random_state = 2, stratify = y)
from sklearn.tree import DecisionTreeClassifier

drugTree = DecisionTreeClassifier(criterion = "entropy",max_depth = 4)

drugTree
print("Training Set: ", X_train.shape, y_train.shape)

print("Testing Set: ", X_test.shape, y_test.shape)
drugTree.fit(X_train,y_train)
predTree = drugTree.predict(X_test)

print(predTree[0:5])

print(y_test[0:5])
from sklearn import metrics

print("The Accuracy Score is:", metrics.accuracy_score(y_test,predTree))
data_feature_names = ['Age', 'Sex', 'BP', 'Cholesterol','Na_to_K']


dot_data = tree.export_graphviz(drugTree,

                                feature_names=data_feature_names,

                                out_file=None,

                                filled=True,

                                rounded=True)

graph = pydotplus.graph_from_dot_data(dot_data)



colors = ('turquoise', 'orange')

edges = collections.defaultdict(list)



for edge in graph.get_edge_list():

    edges[edge.get_source()].append(int(edge.get_destination()))



for edge in edges:

    edges[edge].sort()    

    for i in range(2):

        dest = graph.get_node(str(edges[edge][i]))[0]

        dest.set_fillcolor(colors[i])

filename = "tree.png"

graph.write_png(filename)

img = mpimg.imread(filename)

plt.figure(figsize=(100,200))

plt.imshow(img,interpolation = 'nearest')

plt.show()