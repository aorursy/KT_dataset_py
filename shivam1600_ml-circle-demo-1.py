from sklearn.datasets import load_iris
from sklearn import tree
iris = load_iris()

print("Feature Names :", iris.feature_names)
print("Target Names :", iris.target_names)
print("Some Rows form Dataset :")
for i in [1,56, 140]:
    print(iris.data[i], iris.target_names[iris.target[i]])
clf = tree.DecisionTreeClassifier()
clf = clf.fit(iris.data, iris.target)
iris.target_names[clf.predict([[1,1,1,1],[5,5,5,5]])]
import graphviz
dot_data = tree.export_graphviz(clf, out_file=None, 
                     feature_names=iris.feature_names,  
                     class_names=iris.target_names,  
                     filled=True, rounded=True,  
                     special_characters=True)  
graph = graphviz.Source(dot_data)  
graph 
