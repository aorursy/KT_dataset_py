import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree
iris = load_iris()
test_index = [0,50,100]
#Let's train our data now
train_target = np.delete(iris.target, test_index)
train_data = np.delete(iris.data, test_index, axis=0)
#Let's test it now
test_target = iris.target[test_index]
test_data = iris.data[test_index]
obj = tree.DecisionTreeClassifier()
obj.fit(train_data, train_target)
test_target
obj.predict(test_data)
#Importing tree in pdf
from sklearn.externals.six import StringIO
import pydot
dot_data = StringIO()
tree.export_graphviz(obj, out_file = dot_data, feature_names = iris.feature_names,

                     class_names = iris.target_names,

                     filled = True,

                     rounded = True,

                     impurity = False)
graph = pydot.graph_from_dot_data(dot_data.getvalue())

#graph.write_pdf("iris.pdf")