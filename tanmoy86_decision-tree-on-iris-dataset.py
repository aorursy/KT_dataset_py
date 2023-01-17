import sklearn.datasets as datasets

import pandas as pd



from sklearn.externals.six import StringIO  

from IPython.display import Image  

from sklearn.tree import export_graphviz

import pydotplus



from sklearn.tree import DecisionTreeClassifier



iris=datasets.load_iris()

df=pd.DataFrame(iris.data, columns=iris.feature_names)

df.head(60)

y=iris.target
dtree=DecisionTreeClassifier()

dtree.fit(df,y)
dot_data = StringIO()

export_graphviz(dtree, out_file=dot_data,  

                filled=True, rounded=True,

                special_characters=True)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  

Image(graph.create_png())