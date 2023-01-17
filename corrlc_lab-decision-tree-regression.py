import pandas as pd

import graphviz



from sklearn import tree

from sklearn.tree import export_graphviz
data = pd.read_csv("../input/cars_resale_value.csv")

data.head()
feature_cols = ['brand','mileage']

X = data[feature_cols]

y = data['resale_value']
dtr = tree.DecisionTreeRegressor()

dtr = dtr.fit(X,y)
dot_data = export_graphviz(dtr, out_file=None, 

                      feature_names=['brand', 'resale'],  

                      filled=True, rounded=True,  

                      special_characters=True)  

graph = graphviz.Source(dot_data)
graph