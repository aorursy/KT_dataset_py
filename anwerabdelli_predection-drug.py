import numpy as np 

import pandas as pd

from sklearn.tree import DecisionTreeClassifier

from sklearn import metrics

import matplotlib.pyplot as plt
my_data=pd.read_csv("../input/drug200.csv")

my_data[0:5]
my_data.shape
#Feature Matrix:

X=my_data[["Age","Sex","BP","Cholesterol","Na_to_K"]].values

X[0:5]
from sklearn import preprocessing

le_sex = preprocessing.LabelEncoder()

le_sex.fit(['F','M'])

X[:,1] = le_sex.transform(X[:,1]) 





le_BP = preprocessing.LabelEncoder()

le_BP.fit([ 'LOW', 'NORMAL', 'HIGH'])

X[:,2] = le_BP.transform(X[:,2])





le_Chol = preprocessing.LabelEncoder()

le_Chol.fit([ 'NORMAL', 'HIGH'])

X[:,3] = le_Chol.transform(X[:,3]) 



X[0:5]
#Target Variable:

y=my_data["Drug"]

y[0:5]
from sklearn.model_selection import train_test_split
X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=3)
print("Shape of X_trainset: ",X_trainset.shape)

print("Shape of y_trainset: ",y_trainset.shape)
print("Shape of X_testset: ",X_testset.shape)

print("Shape of y_testset: ",y_testset.shape)
drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)

drugTree # it shows the default parameters
#We will fit the data with the training feature matrix X_trainset and training response vector y_trainset

drugTree.fit(X_trainset,y_trainset)
predTree=drugTree.predict(X_testset)
print("Predictions : ",predTree[0:5])

print("Actual Values : \n",y_testset[0:5])
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_testset, predTree))
from sklearn.externals.six import StringIO

#import pydotplus

import matplotlib.image as mpimg

from sklearn import tree

%matplotlib inline 
#with open("drugTree.dot", "w") as f:

    #f = tree.export_graphviz(, out_file=f)

import graphviz



data = export_graphviz(drugTree,out_file=None,filled=True,rounded=True,special_characters=True)

graph = graphviz.Source(data)

graph