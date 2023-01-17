from IPython.display import Image

Image(filename ="../input/dszdecisiontree/min-samples-split.png", width=500, height=500)
import pandas as pd

import numpy as np
df = pd.read_csv("../input//dszdecisiontree/bank-numeric.txt")
df.head()
features = df.drop("deposit_cat", axis=1)

targets = df['deposit_cat']
features_names = list(features.columns)

targets_names = ["Deposit_cat"]
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
X_train, X_test, y_train, y_test = train_test_split(features, targets, random_state=0)
dt = DecisionTreeClassifier()
model = dt.fit(X_train, y_train)
predictions = model.predict(X_test)
from sklearn import tree
import pydot

import graphviz
dot_data = tree.export_graphviz(

         dt, 

         out_file=None,

         feature_names=features_names,

         filled=True, rounded=True,

         proportion=False,

         node_ids=True,

         rotate=False

        )  

graph = graphviz.Source(dot_data)  

graph
dt = DecisionTreeClassifier(max_depth=6)



model = dt.fit(X_train, y_train)



predictions = model.predict(X_test)
dot_data = tree.export_graphviz(

         dt, 

         out_file=None,

         feature_names=features_names,

         filled=True, rounded=True,

         proportion=False,

         node_ids=True,

         rotate=False

        )  

graph = graphviz.Source(dot_data)  

graph
from mlxtend.plotting import plot_decision_regions

import matplotlib.pyplot as plt
def compara_modelos(maxdepth):

    if maxdepth == 0:

        dt = tree.DecisionTreeClassifier(random_state=1)

    else:   

        dt = tree.DecisionTreeClassifier(random_state=1, max_depth=maxdepth)

    dt.fit(X_train, y_train)

    train_score = dt.score(X_train, y_train)

    test_score = dt.score(X_test, y_test)

    return train_score,test_score
print('{:10} {:20} {:20}'.format('depth', 'Training score','Testing score'))

print('{:10} {:20} {:20}'.format('-----', '--------------','-------------'))

depth = np.linspace(15,0,16)



for profundidade in depth:

    if profundidade != 0:

        print('{:1}         {} '.format(profundidade,str(compara_modelos(profundidade))))

    else:

        print('{:1}         {} '.format('Full',str(compara_modelos(profundidade))))

    