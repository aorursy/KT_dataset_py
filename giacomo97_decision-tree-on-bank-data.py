# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import tree

from sklearn.metrics import accuracy_score

import graphviz



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data=pd.read_csv("/kaggle/input/bankbalanced/bank.csv")

data.head()
from sklearn import preprocessing



le = preprocessing.LabelEncoder()

for i in range(0,17):

    data.iloc[:,i] = le.fit_transform(data.iloc[:,i])

data.head()
nrows=len(data.index)

percentage=round((nrows*70)/100)

trainingData=data.iloc[:percentage,:]

testData=data.iloc[percentage:,:]



print("Number of training data examples "+str(len(trainingData.index)))

print("Number of test examples "+str(len(testData.index)))

train_x=trainingData[["job","marital","education","balance","housing"]]

train_y=trainingData["loan"]



test_x=testData[["job","marital","education","balance","housing"]]

test_y=testData["loan"]



train_x.head()



featureNames=["Job","marital","education","balance","housing"]

classNames=["no","yes"]
treeClassifier=tree.DecisionTreeClassifier(criterion="entropy")

treeClassifier.fit(train_x,train_y)



prediction = treeClassifier.predict(train_x)

accuracyScore = accuracy_score(train_y, prediction)

print("The trained tree has an accuracy score on the train set of "+str(accuracyScore))



prediction = treeClassifier.predict(test_x)

accuracyScore = accuracy_score(test_y, prediction)

print("The trained tree has an accuracy score on the test set of "+str(accuracyScore))



dot_data = tree.export_graphviz(decision_tree=treeClassifier, feature_names=featureNames, class_names=classNames,filled=True,rounded=True, max_depth=4)

graph = graphviz.Source(dot_data) 

graph
for i in range(3,11):

    treeClassifier=tree.DecisionTreeClassifier(criterion="entropy",max_depth=i)

    treeClassifier.fit(train_x,train_y)



    prediction = treeClassifier.predict(train_x)

    accuracyScore = accuracy_score(train_y, prediction)

    print("The trained tree with depth "+str(i)+" has an accuracy score on the train set of "+str(accuracyScore))



    prediction = treeClassifier.predict(test_x)

    accuracyScore = accuracy_score(test_y, prediction)

    print("The trained tree with depth "+str(i)+" has an accuracy score on the test set of "+str(accuracyScore)+"\n")
treeClassifier=tree.DecisionTreeClassifier(criterion="gini")

treeClassifier.fit(train_x,train_y)



prediction = treeClassifier.predict(train_x)

accuracyScore = accuracy_score(train_y, prediction)

print("The trained tree has an accuracy score on the train set of "+str(accuracyScore))



prediction = treeClassifier.predict(test_x)

accuracyScore = accuracy_score(test_y, prediction)

print("The trained tree has an accuracy score on the test set of "+str(accuracyScore))



dot_data = tree.export_graphviz(decision_tree=treeClassifier, feature_names=featureNames, class_names=classNames,filled=True,rounded=True, max_depth=4)

graph = graphviz.Source(dot_data) 

graph
for i in range(3,11):

    treeClassifier=tree.DecisionTreeClassifier(criterion="gini",max_depth=i)

    treeClassifier.fit(train_x,train_y)



    prediction = treeClassifier.predict(train_x)

    accuracyScore = accuracy_score(train_y, prediction)

    print("The trained tree with depth "+str(i)+" has an accuracy score on the train set of "+str(accuracyScore))



    prediction = treeClassifier.predict(test_x)

    accuracyScore = accuracy_score(test_y, prediction)

    print("The trained tree with depth "+str(i)+" has an accuracy score on the test set of "+str(accuracyScore)+"\n")