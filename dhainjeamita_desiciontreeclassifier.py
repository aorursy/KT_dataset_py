from sklearn.datasets import load_iris
from sklearn import tree
import graphviz
import pandas as pd

empDataframe = pd.read_csv('../input/MFGEmployees4.csv')
newDataframe = empDataframe.iloc[0:100]
#print(empDataframe.columns)
#print (empDataframe.head())
#print (empDataframe.describe())
X = newDataframe.iloc[:,9:12].values
Y = newDataframe['Gender'].values
#print (X)
clf = tree.DecisionTreeClassifier()
#clf = clf.fit(iris.data, iris.target)
clf = clf.fit(X, Y)
dot_data = tree.export_graphviz(clf, out_file=None) 
graph = graphviz.Source(dot_data) 
graph.render("iris") 
dot_data = tree.export_graphviz(clf, out_file=None, 
                         feature_names=['Age','Service','AbHours'],  
                        class_names=None,  
                         filled=True, rounded=True)  
graph = graphviz.Source(dot_data)  
graph 




