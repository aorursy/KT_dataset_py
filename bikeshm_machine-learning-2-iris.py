from sklearn.datasets import load_iris
from sklearn import tree
import numpy as np

import pandas as pd
iris = load_iris()   #load irish data set
print("feature_names") #properties
iris.feature_names
print("iris.data", len(iris.data))
print(iris.data)
#jus for understanding data and feature 
#feature_names => heading of the table column
# data  => is the table data

tempData = pd.DataFrame(data= np.c_[iris['data']],columns= iris['feature_names'])
tempData.head(10) # printing first 10 items 
print("target_names") #irish type name
iris.target_names
print("iris.target", len(iris.target))
iris.target
print("Target index, target name and data ")
print("--------------------------------------")

for i in range(len(iris.target)) :
    print("row : %d | label : %s - %s | feature : %s" % (i, iris.target[i], iris.target_names[ iris.target[i] ] , iris.data[i]))
#Full data details visualisation 
df = pd.DataFrame(np.column_stack((iris.data, iris.target)), columns = iris.feature_names+['target'])
df['species']  = df.target.replace(dict(enumerate(iris.target_names)))
df

#0 = first elemt in array (first:'setosa'),  50 first 'versicolor', 100 first 'virginica']
test_idx = [0, 50, 100]
#training data
train_target = np.delete(iris.target, test_idx)  #deleting 3 entry from targets
train_data = np.delete(iris.data, test_idx, axis=0) #deleting 3 entry from data

print("train_target" , len(train_target))
train_target
print("train_data", len(train_data))
train_data
#testing data
test_target = iris.target[test_idx] #test_idx = [0, 50, 100]   ==> 0 = first elemt in array (first:'setosa'),  50 first 'versicolor', 100 first 'virginica'] 
test_data = iris.data[test_idx]
print("test_target")
test_target
print("test_data", len(test_data))
test_data  # input for prediction
clf = tree.DecisionTreeClassifier()

clf.fit(train_data, train_target )
result = clf.predict(test_data)
print("Prediction result : ")
print(result)
print("Prediction result explanation : %s "% result)
for i in range(len(test_data)) :
    print(" feature : %s | prediction : %s - Target name: %s" % ( test_data[i] , result[i], iris.target_names[ result[i] ]))
print(test_data[0], test_target[0])
import graphviz

dot_data = tree.export_graphviz(clf, out_file=None)
graph = graphviz.Source(dot_data)
graph.render("iris")
dot_data = tree.export_graphviz(clf, out_file=None,
                         feature_names=iris.feature_names,
                         class_names=iris.target_names,
                         filled=True, rounded=True,
                         special_characters=True)
graph = graphviz.Source(dot_data)
graph