# importing libraries

from sklearn.datasets import load_iris

from sklearn import tree

import graphviz
# importing dataset

iris = load_iris()



# iris data

print(iris.data[:5])



# target

print(iris.target[:5])
# instantiating model

clf = tree.DecisionTreeClassifier()



# fitting the model

clf = clf.fit(iris.data, iris.target)
# plotting classification tree

tree.plot_tree(clf.fit(iris.data, iris.target))
dot_data = tree.export_graphviz(clf, out_file=None) 

graph = graphviz.Source(dot_data) 

graph.render("iris") # name of the file to which you want save
dot_data = tree.export_graphviz(clf, # classification model that need to be plotted

                                out_file=None, 

                                feature_names=iris.feature_names, # feature names / names of the dataframe column (df.columns)

                                class_names=iris.target_names, # target label / name

                                filled=True, rounded=True,  

                                special_characters=True)  

graph = graphviz.Source(dot_data)  

graph 