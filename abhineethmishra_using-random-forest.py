# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train_set=pd.read_csv("../input/train/train.csv")
train_set.head()
train_set.describe()
lst=train_set["Type"].tolist()
dog=0
cat=0
for i in lst:
    if lst[i]==1:
        dog+=1
    else:
        cat+=1
print("Dog:"+str(dog)+"Cat:"+str(cat))
#no of dogs=6861,cats=8132
#divide into output and input
X_train=train_set.drop(columns=["AdoptionSpeed"])
y_train=train_set.AdoptionSpeed
X_train
#import matplotlib.pyplot as plt
#creating bins for data
#Agelst=X_train["Age"].tolist()
#bins=[0,1,2,3,4,5]
#binning age [0-6,6-12,12-36,36-60,60-96,96-255]
#binwidth=int((max(X_train["Age"])-min(X_train["Age"]))/6)
#bins=range(min(X_train["Age"]),max(X_train["Age"]),binwidth)
#group_names=[0,1,2,3,4,5]
#X_train["Age binned"]=pd.cut(X_train["Age"],bins,labels=group_names)
l=X_train["Age"].tolist()
newlst=[]
for i in l: 
    if i>=0 and i<=6:
        newlst.append(0)
    elif i>=7 and i<=12:
        newlst.append(1)
    elif i>=13 and i<=36:
        newlst.append(2)
    elif i>=37 and i<=60:
        newlst.append(3)
    elif i>=61 and i<=96:
        newlst.append(4)
    elif i>=97:
        newlst.append(5)
#X_train=X_train.drop(columns=["Age binned"])
X_train["Age binned"]=newlst
X_train=X_train.drop(columns=["Age"])
X_train
#binning fee
feelst=[]
l=X_train["Fee"].tolist()
for i in l:
    if i==0:
        feelst.append(0)
    elif i>=1 and i<=25:
        feelst.append(1)
    elif i>=26 and i<=60:
        feelst.append(2)
    elif i>=61 and i<=108:
        feelst.append(3)
    elif i>=109 and i<=210:
        feelst.append(4)
    elif i>=211:
        feelst.append(5)
#X_train["Fees"]=X_train["Fee"]
X_train["Fee binned"]=feelst
X_train=X_train.drop(columns=["Fee"])
#import statistics
#X_train["Fees"]=(X_train["Fee"]-(sum(X_train.Fee.tolist())/len(X_train.Fee.tolist())))/statistics.stdev(X_train.Fee.tolist())
#converting description using NLP
from textblob import TextBlob
desc=X_train["Description"].tolist()
l=[]
for i in range(len(desc)):
    l.append(TextBlob(str(desc[i])).sentiment.polarity)
X_train["Description Polarity"]=l
X_train=X_train.drop(columns=["Description","RescuerID","Name","PetID"])
#one hot encoding state names
from sklearn.preprocessing import LabelBinarizer
lb = LabelBinarizer()
#X_train['States'] = lb.fit_transform(X_train['State']).tolist()
X_train = pd.concat([X_train, pd.get_dummies(X_train['State'])], axis=1)
X_train=X_train.drop(columns=["State"])
X_train
#getting correlations
X_train.join(y_train).corr()["AdoptionSpeed"].sort_values(ascending=False)
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
X_train=X_train.drop(columns=[41415])
xTrain, xTest, yTrain, yTest = train_test_split(X_train, y_train, test_size = 0.2, random_state = 1)
clf = RandomForestClassifier()
#X_test=X_test.drop(columns=[41415])
clf.fit(xTrain,yTrain)
xTest
from sklearn.metrics import accuracy_score
score = accuracy_score(yTest,clf.predict(xTest))
print(score)

#from sklearn.cross_validation import cross_val_score
#cross_val=np.mean(cross_val_score(clf, xTrain, yTrain, cv=10)).astype(str)
#print("Cross Validation Accuracy"+cross_val)
grid_param = {  
    'n_estimators': [100, 300, 500, 800, 1000],
    'criterion': ['gini', 'entropy'],
    'bootstrap': [True, False]
}


#from sklearn.model_selection import GridSearchCV
#Finding the best parameters for grid search
#gd_sr = GridSearchCV(estimator=clf,  
#                     param_grid=grid_param,
#                    scoring='accuracy',
 #                    cv=5,
 #                    n_jobs=-1)
#gd_sr.fit(xTrain, yTrain)  
#best_parameters = gd_sr.best_params_ 
#best_parameters={'bootstrap': True, 'criterion': 'entropy', 'n_estimators': 100} 
#print('best parameters:')
#print(best_parameters)
#on running best params are {'bootstrap': True, 'criterion': 'entropy', 'n_estimators': 500}
params={'bootstrap': True, 'criterion': 'entropy', 'n_estimators': 500}
clf=RandomForestClassifier(**params)
clf.fit(xTrain,yTrain)
score = accuracy_score(yTest,clf.predict(xTest))
print(score)
#create submission csv
#get test csv into a df
testdf=pd.read_csv('../input/test/test.csv')
testdf
#changing the test set
l=testdf["Age"].tolist()
newlst=[]
for i in l: 
    if i>=0 and i<=6:
        newlst.append(0)
    elif i>=7 and i<=12:
        newlst.append(1)
    elif i>=13 and i<=36:
        newlst.append(2)
    elif i>=37 and i<=60:
        newlst.append(3)
    elif i>=61 and i<=96:
        newlst.append(4)
    elif i>=97:
        newlst.append(5)

testdf["Age binned"]=newlst
testdf=testdf.drop(columns=["Age"])

#binning fee
feelst=[]
l=testdf["Fee"].tolist()
for i in l:
    if i==0:
        feelst.append(0)
    elif i>=1 and i<=25:
        feelst.append(1)
    elif i>=26 and i<=60:
        feelst.append(2)
    elif i>=61 and i<=108:
        feelst.append(3)
    elif i>=109 and i<=210:
        feelst.append(4)
    elif i>=211:
        feelst.append(5)
#X_train["Fees"]=X_train["Fee"]
testdf["Fee binned"]=feelst
testdf=testdf.drop(columns=["Fee"])

#converting description using NLP
from textblob import TextBlob
desc=testdf["Description"].tolist()
l=[]
for i in range(len(desc)):
    l.append(TextBlob(str(desc[i])).sentiment.polarity)
testdf["Description Polarity"]=l
testdf=testdf.drop(columns=["Description","RescuerID","Name","PetID"])

#one hot encoding state names
from sklearn.preprocessing import LabelBinarizer
lb = LabelBinarizer()
#X_train['States'] = lb.fit_transform(X_train['State']).tolist()
testdf = pd.concat([testdf, pd.get_dummies(testdf['State'])], axis=1)
testdf=testdf.drop(columns=["State"])

temp=pd.read_csv('../input/test/test.csv')
ansdf=pd.DataFrame(temp["PetID"])
ansdf['AdoptionSpeed']=clf.predict(np.array(testdf))
import csv
#csv=ansdf.to_csv(encoding='utf-8', sep=',',index=False)
ansdf.to_csv('submission.csv',index=False)
