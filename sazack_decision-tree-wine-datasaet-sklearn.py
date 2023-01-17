import sklearn
from sklearn.datasets import load_wine
import numpy as np
from  sklearn.model_selection import train_test_split
from  sklearn.model_selection import ShuffleSplit

dataset = load_wine()

# print(dataset.DESCR)
X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.30, random_state = 45)


from sklearn import tree
classifier = tree.DecisionTreeClassifier()
classifier = classifier.fit(X_train,y_train)

prediction = classifier.predict(X_test)



accuracy = sklearn.metrics.accuracy_score(prediction,y_test)
print("Accuracy: ", '%.2f'% (accuracy*100),"%")

#Generating Graph of our Data on Graphviz
#Run this for graph if you are running the notebook on your local device

# import graphviz
# %matplotlib inline
# dotdata = tree.export_graphviz(classifier, out_file=None, feature_names =dataset.feature_names, class_names = dataset.target_names, filled=True, rounded = True, special_characters = True)
# graph = graphviz.Source(dotdata).view()
# graph.render("wine")
#Random Forest Classifier

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 30, max_depth=2)
classifier.fit(X_train,y_train)
prediction = classifier.predict(X_test)

accuracy = sklearn.metrics.accuracy_score(prediction,y_test)
print("Accuracy: ", '%.2f'% (accuracy*100),"%")

#creating our own dummy Random Forest Classifier
randomshuffle = ShuffleSplit(n_splits=7, test_size=0.33, random_state=0)
randomshuffle.get_n_splits(dataset.data)
trees =[]
for train_index, test_index in randomshuffle.split(dataset.data):
    i=0
    i +=1
#     print("Train:",train_index,"Test:",test_index)
    classifier = tree.DecisionTreeClassifier().fit(dataset.data[train_index],dataset.target[train_index])
    predict = classifier.predict(dataset.data[test_index])
    dic={}
    accuracy = sklearn.metrics.accuracy_score(predict,dataset.target[test_index])
#     dic = globals()['sample%s' %i ] = accuracy *100
#     print(dic)
    dic[classifier] = accuracy*100
    trees.append(dic)
#A dictionary file output of each tree and the subsequent Accuracy score of the preidction
print(trees)
    
    
    