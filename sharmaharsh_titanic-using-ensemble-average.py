# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
from sklearn import svm
import graphviz
import csv
# Any results you write to the current directory are saved as output.
train=pd.read_csv("../input/train.csv")
train
test=pd.read_csv("../input/test.csv")


## Passenger ID is useless
# survived is to be predicted
# Class of passenger shows rich poor
# Sex to be sorted to binary

# Count of male survivors
print ("Total people on board",train.count()[0])
print ("Females on board=",train['Sex'][train['Sex']=='female'].count())
print ("Males on board=",train['Sex'][train['Sex']=='male'].count())
print ("%males survived =",train['Survived'][train['Survived']==1][train['Sex']=='male'].count()*1.0/train['Survived'][train['Sex']=='male'].count())
print ("%females survived =",train['Survived'][train['Survived']==1][train['Sex']=='female'].count()*1.0/train['Survived'][train['Sex']=='female'].count())
## more %females survived
#Making Sex of person a binary choice 
train['Gender']=train['Sex'].str.contains('female').apply(lambda x:0 if x==False else 1) ##females are 1 males are 0
train['Gender']
train.drop('Sex',inplace=True,axis=1)
test['Gender']=test['Sex'].str.contains('female').apply(lambda x:0 if x==False else 1) ##females are 1 males are 0
test['Gender']
test.drop('Sex',inplace=True,axis=1)
print (train['Embarked'].value_counts()) ## give instances and occurance of each of them
## most people climbed from S>C>Q
for s in train['Embarked'].unique():
    print ("embark from",s)
    print ("%life rate=",train["Embarked"][train["Embarked"]==s][train['Survived']==1].count()/train["Embarked"][train["Embarked"]==s].count())
    #people from C>Q>S survival
train['Embark']=train['Embarked'].apply(lambda x:0 if x=='S' else (1 if x=='C' else 2)) ## making value as number to help apply regression and stuff easier
print (train['Embark'])
train.drop('Embarked',inplace=True,axis=1) #drops the embarked axis
test['Embark']=test['Embarked'].apply(lambda x:2 if x=='Q' else (1 if x=='C' else 0)) ## making value as number to help apply regression and stuff easier
print (test['Embark'])
test.drop('Embarked',inplace=True,axis=1) #drops the embarked axis
for s in train['Pclass'].unique():
    print ("Count of", s,"is",train['Pclass'][train['Pclass']==s].count())
    print ("Survival rate of passenger class ",s, "is",train['Survived'][train["Pclass"]==s][train['Survived']==1].count()/train['Survived'][train["Pclass"]==s].count())
## survival rate of 1>2>3 rich survived more than poor, who knows?
print (test[np.isnan(test['SibSp'])==True])
print (test[np.isnan(test['Parch'])==True]) 
## no number is nan
train['Fam']=train['SibSp']+train['Parch']
train['Fam'] #count of family
test['Fam']=test['SibSp']+test['Parch']
test['Fam'] #count of family
#pd.value_counts(train['Survived']).plot.bar()
X_train, X_test, y_train, y_test = train_test_split( train.drop('Survived',axis=1), train['Survived'], test_size=0.33, random_state=42)
from sklearn import tree
Dtree=tree.DecisionTreeClassifier()
feature_used=['Gender','Fam','Pclass','Embark']
Dtree=Dtree.fit(X_train[feature_used].values,y_train.values)
RandomForest=RandomForestClassifier()
RandomForest.fit(X_train[feature_used].values,y_train.values)
Percept=Perceptron()
Percept.fit(X_train[feature_used].values,y_train.values)
logreg=LogisticRegression()
logreg.fit(X_train[feature_used].values,y_train.values)
svmtrain=svm.SVC()
svmtrain.fit(X_train[feature_used].values,y_train.values)
import graphviz
#with open("./tree1","w") as f:
    #f = tree.export_graphviz(Dtree, out_file=f)
dot_data = tree.export_graphviz(Dtree, out_file=None,)
                         #feature_names=iris.feature_names,  
                         #class_names=iris.target_names,  
                         #filled=True, rounded=True,  
                         #special_characters=True)  
graph = graphviz.Source(dot_data)  
graph 
print ("Tree accuracy=",Dtree.score(X_test[feature_used],y_test))
print ("Random Forest accuracy=",RandomForest.score(X_test[feature_used],y_test))
print ("Perceptron Accuracy=",Percept.score(X_test[feature_used],y_test))
print ("Logistic Regression Accuracy=", logreg.score(X_test[feature_used],y_test))
print ("SVM accuracy=", svmtrain.score(X_test[feature_used],y_test))
Tree_predict=Dtree.predict(test[feature_used])
RDM_predict=RandomForest.predict(test[feature_used])
Percept_predict=Percept.predict(test[feature_used])
logreg_predict=logreg.predict(test[feature_used])
svm_predict=svmtrain.predict(test[feature_used])
with open("Treeout.csv","w") as file:
    writer=csv.writer(file,delimiter=",")
    writer.writerow(("PassengerId","Survived"))
    for i in range(test.shape[0]):
        writer.writerow([test.PassengerId.values[i],Tree_predict[i]])
with open("RandomTree.csv","w") as file:
    writer=csv.writer(file,delimiter=",")
    writer.writerow(("PassengerId","Survived"))
    for i in range(test.shape[0]):
        writer.writerow([test.PassengerId.values[i],RDM_predict[i]])
with open("Perceptout.csv","w") as file:
    writer=csv.writer(file,delimiter=",")
    writer.writerow(("PassengerId","Survived"))
    for i in range(test.shape[0]):
        writer.writerow([test.PassengerId.values[i],Percept_predict[i]])
with open("LogRegout.csv","w") as file:
    writer=csv.writer(file,delimiter=",")
    writer.writerow(("PassengerId","Survived"))
    for i in range(test.shape[0]):
        writer.writerow([test.PassengerId.values[i],logreg_predict[i]])
with open("SVMout.csv","w") as file:
    writer=csv.writer(file,delimiter=",")
    writer.writerow(("PassengerId","Survived"))
    for i in range(test.shape[0]):
        writer.writerow([test.PassengerId.values[i],svm_predict[i]])
mix_predict=logreg_predict+RDM_predict+Tree_predict+svm_predict
mix_predict[mix_predict<2]=0
mix_predict[mix_predict>=2]=1
with open("mixout.csv","w") as file:
    writer=csv.writer(file,delimiter=",")
    writer.writerow(("PassengerId","Survived"))
    for i in range(test.shape[0]):
        writer.writerow([test.PassengerId.values[i],mix_predict[i]])
