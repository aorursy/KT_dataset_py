# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train=pd.read_csv('/kaggle/input/titanic/train.csv')

test=pd.read_csv('/kaggle/input/titanic/test.csv')

testtrain=[train,test]
def feSex():

    sexMap={"male":1,"female":0}

    for row in testtrain:

        row['Sex']=row['Sex'].map(sexMap)

    print("Sex-FEd")
def feEmbark():

    embarkMap={"S":0,"C":1,"Q":2}

    for row in testtrain:

        row['Embarked']=row['Embarked'].map(embarkMap)

    print("Embarked-FEd")
def feName():

    for row in testtrain:

        row['Title']=row['Name'].str.extract(' ([A-Za-z]+)\.',expand=False)

    print("Name-TitleExtracted")
def feTitle():

    titleMap={

        'Mr':0,'Miss':1,'Mrs':2,

        'Master':3,'Dr':3,'Rev':3,'Major':3,'Mlle':3,'Col':3,'Don':3,'Dona':3,

        'Jonkheer':3,'Countess':3,'Ms':3,'Sir':3,'Capt':3,'Mme':3,'Lady':3

    }

    

    for row in testtrain:

        row['Title']=row['Title'].map(titleMap)

    print("Title-FEd")   
def fillAge():

    train['Age'].fillna(train.groupby('Title')['Age'].transform('median'),inplace=True)

    test['Age'].fillna(train.groupby('Title')['Age'].transform('median'),inplace=True)

    print("Age filled : trainNA: "+str(train['Age'].isna().sum()))

    print("Age filled :testNA: "+str(test['Age'].isna().sum()))

    
def feAge():

    for row in testtrain:

        row.loc[row.Age<=16,'Age']=0,

        row.loc[(row.Age>16) & (row.Age<=26),'Age']=1

        row.loc[(row.Age>26) & (row.Age<=36),'Age']=2

        row.loc[(row.Age>36) & (row.Age<=60),'Age']=3

        row.loc[row.Age>60,'Age']=4

    print("Age-FEd")
#Majority is 'S' missing is only rows hence going with 'S'



def fillEmbarked():

    train['Embarked'].fillna('S',inplace=True)

    test['Embarked'].fillna('S',inplace=True)

    print("Embarked filled : trainNA: "+str(train['Embarked'].isna().sum()))

    print("Embarked filled : testNA: "+str(test['Embarked'].isna().sum()))

    
def dropColumns():

    columnsToBeDropped=['Name','Ticket','Fare','Cabin']

    train.drop(columnsToBeDropped,axis=1,inplace=True)

    test.drop(columnsToBeDropped,axis=1,inplace=True)

    train.drop(['PassengerId'],axis=1,inplace=True)

    print("Columns Dropped")

    
def applyFE():    

    feSex()

    fillEmbarked()

    feEmbark()

    feName()

    feTitle()

    fillAge()

    feAge()    

    dropColumns()
applyFE()
train.head()
def createTrainAndTarget():

    train_data=train.drop(['Survived'],axis=1)

    target=train['Survived']

    return train_data,target
train_data,target=createTrainAndTarget()

train_data.head()
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC



import numpy as np
from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

kfold=KFold(n_splits=10,shuffle=True,random_state=0)
clf=DecisionTreeClassifier()

scoring='accuracy'

score=cross_val_score(clf,train_data,target,cv=kfold,n_jobs=1,scoring=scoring)

print("DecisionTreeClassifier Scores : "+str(score))

print("DecisionTreeClassifier MeanScore : "+str(score.mean()*100))
clf=RandomForestClassifier(n_estimators=10)

scoring="accuracy"

score=cross_val_score(clf,train_data,target,cv=kfold,scoring=scoring)

print("RandomForestClassifier Scores : "+str(score))

print("RandomForestClassifier MeanScore : "+str(score.mean()*100))
clf=GaussianNB()

scoring='accuracy'

score=cross_val_score(clf,train_data,target,cv=kfold,scoring=scoring)

print("GaussianNB Scores : "+str(score))

print("GaussianNB MeanScore : "+str(score.mean()*100))
clf=SVC(gamma='auto')

score=cross_val_score(clf,train_data,target,cv=kfold,scoring="accuracy")

print("SVC Scores : "+str(score))

print("SVC MeanScore : "+str(score.mean()*100))

clf=SVC(gamma='auto')

clf.fit(train_data,target)

test_data=test.drop(['PassengerId'],axis=1).copy()

prediction=clf.predict(test_data)
submission=pd.DataFrame({

    "PassengerId":test['PassengerId'],

    "Survived":prediction

})

submission.to_csv('submission.csv', index=False)