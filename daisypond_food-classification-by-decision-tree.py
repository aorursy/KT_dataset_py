!pip install pydotplus



import pandas as pd

import numpy as np

from sklearn.cluster import KMeans

import sklearn.preprocessing as sp

import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier

from IPython.display import Image, display_png

from pydotplus import graph_from_dot_data

from sklearn.tree import export_graphviz
df = pd.read_csv('../input/Emoji Diet Nutritional Data (g) - EmojiFoods (g).csv')

df
labels =[0,0,0,0,0,0,0,0,0,0,

         0,0,0,0,1,0,1,1,1,1,

         1,1,1,1,1,2,2,2,2,3,

         3,3,3,4,4,4,4,4,4,4,

         2,2,2,3,5,5,5,5,5,5,

         5,5,6,6,6,6,6,6]

len(labels)



df['labels']= labels

df.info()
tree = DecisionTreeClassifier(criterion='gini', random_state =1, min_samples_leaf=1)



X_train = df.iloc[:, 2:11]

y_train = df['labels']



tree.fit(X_train, y_train)

dot_data = export_graphviz(tree, filled = True, rounded = True, class_names = ['fruits','vegetables', 'grain crops', 'animals/fishes', 'junk foods', 'desserts', 'drinks'],

                          feature_names = df.columns[2:11].values, out_file = None)



graph = graph_from_dot_data(dot_data)

graph.write_png('tree.png')

display_png(Image('tree.png'))