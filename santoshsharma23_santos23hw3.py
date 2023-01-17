#Please note that task 4 is done by hand as well, will be verified by code here

#First question 1 of task 4 is done
import pandas as pd

Task4First = pd.read_csv("../input/task4first/Task4First.csv")
# Importing the dataset

dataset = Task4First

Xtrain=dataset[['Home/Away','In/Out','Media']]

Ytrain=dataset[['Label']]
Xtrain
Ytrain
Xtrain = pd.get_dummies(Xtrain) #Alternative way to do one hot encoding (OHE)
Xtrain
from sklearn import tree

clf=tree.DecisionTreeClassifier(random_state=0,criterion="entropy")

clf.fit(Xtrain, Ytrain)
from graphviz import Source

#Source( tree.export_graphviz(clf, out_file=None))

Source( tree.export_graphviz(clf, out_file=None, feature_names=Xtrain.columns,class_names=['Lose','Win']))
#So far question 1 of task 4 is verified
#Now question 2 of task 4
import pandas as pd
dataset = pd.read_csv("../input/task4second/Task4.csv")
dataset
X=dataset[['Outlook','Temperature','Humidity','Windy']]

Y=dataset[['Play']]
XOHE = pd.get_dummies(X) #Alternative way to do one hot encoding (OHE)
XOHE
Xtrain=XOHE.head(14)

Xtest=XOHE.tail(1)

Xtrain
Ytrain=Y.head(14)

#Ytest=Y.tail(1)

Ytrain
from sklearn import tree

clf=tree.DecisionTreeClassifier(random_state=0,criterion="entropy")

clf.fit(Xtrain, Ytrain)
Ypred = clf.predict(Xtest)
Ypred[0]
from graphviz import Source

#Source( tree.export_graphviz(clf, out_file=None))

Source( tree.export_graphviz(clf, out_file=None, feature_names=Xtrain.columns,class_names=['No_Play','Play']))
#So far task 4 is completed, task 4 is performed by hand and code
#Task 5 now will be performed, task 5 is done using only code, please note that for CART, we should use gini index and for ID3 we use entropy
import pandas as pd

Task5 = pd.read_csv("../input/task5g/Task5.csv")
dataset=Task5

dataset
X=dataset[['Home/Away','In/Out','Media']]

Y=dataset[['Play']]
XOHE = pd.get_dummies(X) #Alternative way to do one hot encoding (OHE)
XOHE
Xtrain=XOHE.head(24)

Xtest=XOHE.tail(12)

Xtest
Ytrain=Y.head(24)

Ytrue=Y.tail(12)

Ytrain
from sklearn import tree

clf=tree.DecisionTreeClassifier(random_state=0,criterion="entropy")

clf.fit(Xtrain, Ytrain)
from graphviz import Source

#Source( tree.export_graphviz(clf, out_file=None))

Source( tree.export_graphviz(clf, out_file=None, feature_names=Xtrain.columns,class_names=['Lose','Win']))
Ypred = clf.predict(Xtest)
Ypred=pd.DataFrame(Ypred)
Ypred
Ytrue
from sklearn.metrics import accuracy_score

accuracy_score(Ytrue, Ypred)
from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()

Ytrue = lb_make.fit_transform(Ytrue)

Ypred = lb_make.fit_transform(Ypred)
from sklearn.metrics import precision_score

precision_score(Ytrue, Ypred)
from sklearn.metrics import recall_score

recall_score(Ytrue, Ypred)
from sklearn.metrics import f1_score

f1_score(Ytrue, Ypred)