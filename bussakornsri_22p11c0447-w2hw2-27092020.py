import math
import numpy as np # linear algebra
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
kf = KFold(n_splits=5)
train = pd.read_csv("/kaggle/input/titanic/train.csv")
df = pd.DataFrame(train)
df.head(10)
df['Embarked'].value_counts()
embark = pd.get_dummies(df['Embarked'])
embark
label = df['Survived']
label
df = df.drop(['PassengerId','Survived','Name','Ticket','Cabin', 'Embarked'], axis=1)
df
df = pd.concat([df,embark], axis=1)
df = df.replace('male',0)
df = df.replace('female',1)
df = df.fillna(math.floor(df['Age'].mean()))
feat = df
feat
trainfeat, testfeat, trainlabel, testlabel = train_test_split(feat, label, test_size=0.33, random_state=42)
decisionTree = DecisionTreeClassifier(random_state=100) 
decisionTree.fit(trainfeat, trainlabel)  
decisionTreePredict = decisionTree.predict(testfeat)
print('Decision Tree')
print('Accuracy: ', accuracy_score(testlabel, decisionTreePredict))
print('Recall: ', recall_score(testlabel, decisionTreePredict))
print('Precision: ', precision_score(testlabel, decisionTreePredict))
print('F1-Score: ', f1_score(testlabel, decisionTreePredict))
print('-----------------------------------------------------------\n')
dtFMeasure = []
for i,(train_index, test_index) in enumerate(kf.split(feat), start=1):
  trainfeat_ = feat.loc[train_index,:]
  trainlabel_ = label.loc[train_index]
  testfeat_ = feat.loc[test_index,:]
  testlabel_ = label.loc[test_index]
  decisionTree.fit(trainfeat_, trainlabel_)
  decisionTreePredict_ = decisionTree.predict(testfeat_)
  print('Decision Tree K-Fold: ', i)
  print('Accuracy: ', accuracy_score(testlabel_, decisionTreePredict_))
  print(classification_report(testlabel_,decisionTreePredict_), '\n')
  dtFMeasure.append(f1_score(testlabel_, decisionTreePredict_))
print('Average F1-Measure: ', sum(dtFMeasure)/5)
naiveBayes = GaussianNB() 
naiveBayes.fit(trainfeat, trainlabel)  
naiveBayesPredict = naiveBayes.predict(testfeat)
print('Naive Bayes')
print('Accuracy: ', accuracy_score(testlabel, naiveBayesPredict))
print('Recall: ', recall_score(testlabel, naiveBayesPredict))
print('Precision: ', precision_score(testlabel, naiveBayesPredict))
print('F1-Score: ', f1_score(testlabel, naiveBayesPredict))
print('-----------------------------------------------------------\n')
nbFMeasure = []
for i,(train_index, test_index) in enumerate(kf.split(feat), start=1):
  trainfeat_ = feat.loc[train_index,:]
  trainlabel_ = label.loc[train_index]
  testfeat_ = feat.loc[test_index,:]
  testlabel_ = label.loc[test_index]
  naiveBayes.fit(trainfeat_, trainlabel_)
  naiveBayesPredict_ = naiveBayes.predict(testfeat_)
  print('Naive Bayes K-Fold: ', i)
  print('Accuracy: ', accuracy_score(testlabel_, naiveBayesPredict_))
  print(classification_report(testlabel_,naiveBayesPredict_), '\n')
  nbFMeasure.append(f1_score(testlabel_, naiveBayesPredict_))
print('Average F1-Measure: ', sum(nbFMeasure)/5)
neuralNetwork = MLPClassifier(max_iter=500, random_state=100)
neuralNetwork.fit(trainfeat, trainlabel)  
neuralNetworkPredict = neuralNetwork.predict(testfeat)
print('Neural Network')
print('Accuracy: ', accuracy_score(testlabel, neuralNetworkPredict))
print('Recall: ', recall_score(testlabel, neuralNetworkPredict))
print('Precision: ', precision_score(testlabel, neuralNetworkPredict))
print('F1-Score: ', f1_score(testlabel, neuralNetworkPredict))
print('-----------------------------------------------------------\n')
nnFMeasure = []
for i,(train_index, test_index) in enumerate(kf.split(feat), start=1):
  trainfeat_ = feat.loc[train_index,:]
  trainlabel_ = label.loc[train_index]
  testfeat_ = feat.loc[test_index,:]
  testlabel_ = label.loc[test_index]
  neuralNetwork.fit(trainfeat_, trainlabel_)
  neuralNetworkPredict_ = neuralNetwork.predict(testfeat_)
  print('Neural Network K-Fold: ', i)
  print('Accuracy: ', accuracy_score(testlabel_, neuralNetworkPredict_))
  print(classification_report(testlabel_,neuralNetworkPredict_), '\n')
  nnFMeasure.append(f1_score(testlabel_, neuralNetworkPredict_))
print('Average F1-Measure: ', sum(nnFMeasure)/5)