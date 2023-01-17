#import IRIS data from sklearn library. More info about IRIS data https://archive.ics.uci.edu/ml/datasets/iris



#Attribute set (X):



#1. sepal length in cm

#2. sepal width in cm

#3. petal length in cm

#4. petal width in cm



#Class output (Y):

#-- Iris Setosa

#-- Iris Versicolour

#-- Iris Virginica



from sklearn.datasets import load_iris



#import library DecisionTreeClassifier

from sklearn import tree



#create D-Tree classification function

X, y = load_iris(return_X_y=True)

clf = tree.DecisionTreeClassifier()

clf = clf.fit(X, y)
#plot classification result using GraphViz



import graphviz

iris = load_iris()

dot_data = tree.export_graphviz(clf, out_file=None) 

graph = graphviz.Source(dot_data) 

graph.render("iris")



dot_data = tree.export_graphviz(clf, out_file=None, feature_names=iris.feature_names, class_names=iris.target_names, filled=True, rounded=True, special_characters=True)

graph = graphviz.Source(dot_data)  

graph 

from sklearn.tree import export_text

r = export_text(clf, feature_names=iris['feature_names'])

print(r)