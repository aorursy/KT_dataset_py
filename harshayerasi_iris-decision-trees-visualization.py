!pip install pydotplus
!pip install graphviz
#importing necessary packages

import pandas as pd

import sklearn.datasets as datasets

from sklearn.tree import DecisionTreeClassifier

from sklearn.externals.six import StringIO  

from IPython.display import Image  

from sklearn.tree import export_graphviz

import pydotplus

import graphviz
#loading the data

iris = datasets.load_iris()

X = pd.DataFrame(iris.data, columns=iris.feature_names)

y = iris.target
X.head()
# Defining and fitting

# Maximum depth can be restricted to avoid overfitting in decission tree

dtree = DecisionTreeClassifier(max_depth=2,random_state=10)

dtree.fit(X,y)
dtree.predict([[5,3,1.5,0.4]])
#Visualizing

dot_data = StringIO()

export_graphviz(dtree, out_file=dot_data,  

                filled=True, # Gives different colors to classes

                feature_names = iris.feature_names,

                class_names=['setosa','versi color','virginca'],

                rounded=True,

                special_characters=True

               )

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  

Image(graph.create_png())