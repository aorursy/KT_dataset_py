import numpy as np
import pandas as pd
import csv as csv
from sklearn import svm
pd.set_option('line_width', 100)
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
train.head(10)
train = train.drop(['PassengerId','Name','Ticket','Cabin'],axis=1)
train.head(10)
train.head()
def preprocess(data):
    data.Age = data.Age.fillna(data.Age.median())
    data.Fare = data.Fare.fillna(data.Fare.median())
    data.Embarked = data.Embarked.fillna("S")
    data["Sex"] = data["Sex"].map({'female':0,'male':1}).astype(int)
    data["Embarked"] = data["Embarked"].map({'S':0,'C':1,'Q':2}).astype(int)
    del data['Name']
    del data['Ticket']
    del data['Cabin']
    return data
    
def main():
    train_pre = preprocess(train)
    label = train_pre.iloc[:,1]
    features_train = train_pre.iloc[:,2]
    
    clf = svm.SVC()
    clf.fit(features_train,label)
    
    test_pre = preprocess(test)
    features_test = test_pre.iloc[:,1]
    
    prediction = clf.predict(features_test)
f = open('write1.csv','w')
writer = csv.writer(f)
writer.writerow(["PassengerId","Survived"])
f.close()
