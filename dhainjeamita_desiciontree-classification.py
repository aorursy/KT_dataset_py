# Load libraries
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn import tree
import graphviz
import pandas as pd

column_names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pd.read_csv("../input/iris-dataset/iris.csv", names=column_names)
print ("Shape of the dataset - ")
print (dataset.shape)

print ("Statistical summary of the dataset - ")
print(dataset.describe())

print ("Class Distribution of the dataset - ")
print(dataset.groupby('class').size())
# scatter plot matrix
scatter_matrix(dataset,figsize=(10,10))
pyplot.show()
X = dataset.iloc[:,0:4].values
Y = dataset.iloc[:,4].values
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)
dot_data = tree.export_graphviz(clf, out_file=None) 
graph = graphviz.Source(dot_data) 
graph.render("iris") 
dot_data = tree.export_graphviz(clf, out_file=None, 
                         feature_names=['sepal-length', 'sepal-width', 'petal-length', 'petal-width'],  
                        class_names=None,  
                         filled=True, rounded=True)  
graph = graphviz.Source(dot_data)  
graph 
# Decision Tree using GINI Index
clf = tree.DecisionTreeClassifier(criterion = "gini", random_state = 100, max_depth=3, min_samples_leaf=5)
clf = clf.fit(X, Y)
dot_data = tree.export_graphviz(clf, out_file=None) 
graph = graphviz.Source(dot_data) 
graph.render("iris") 
dot_data = tree.export_graphviz(clf, out_file=None, 
                         feature_names=['sepal-length', 'sepal-width', 'petal-length', 'petal-width'],  
                        class_names=None,  
                         filled=True, rounded=True)  
graph = graphviz.Source(dot_data)  
graph 

# Decision Tree using Entropy Index
clf = tree.DecisionTreeClassifier(criterion = "entropy", random_state = 100, max_depth=10, min_samples_leaf=10)
clf = clf.fit(X, Y)
dot_data = tree.export_graphviz(clf, out_file=None) 
graph = graphviz.Source(dot_data) 
graph.render("iris") 
dot_data = tree.export_graphviz(clf, out_file=None, 
                         feature_names=['sepal-length', 'sepal-width', 'petal-length', 'petal-width'],  
                        class_names=None,  
                         filled=True, rounded=True)  
graph = graphviz.Source(dot_data)  
graph 