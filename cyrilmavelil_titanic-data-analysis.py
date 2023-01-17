import warnings

warnings.filterwarnings("ignore")

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sb

%matplotlib inline

traindata = pd.read_csv("/kaggle/input/titanic/train.csv")

testdata = pd.read_csv("/kaggle/input/titanic/test.csv")

testdat=testdata.PassengerId
sb.heatmap(traindata.isnull())
sb.heatmap(testdata.isnull())
sb.countplot(x='Survived',data=traindata)
sb.countplot(x='Survived',hue='Sex',data=traindata)
sb.countplot(x='Survived',hue='Pclass',data=traindata)
sb.boxplot(x='Survived',y='Age',data=traindata)
sb.boxplot(x='Pclass',y='Age',data=traindata)
def impute_age(cols):

    Age=cols[0]

    Pclass=cols[1]

    

    if pd.isnull(Age):

        if Pclass ==1:

            return 37

        elif Pclass==2:

            return 29

        else:

            return 24  

    else:

        return Age
traindata['Age']=traindata[['Age','Pclass']].apply(impute_age,axis=1)
traindata.drop('Cabin',axis=1,inplace=True)
sb.heatmap(traindata.isnull())
traindata.groupby('Embarked').size()
commonvalues='S'

traindata["Embarked"]=traindata['Embarked'].fillna(commonvalues)

traindata.info()
testdata.drop('Cabin',axis=1,inplace=True)
testdata['Age']=testdata[['Age','Pclass']].apply(impute_age,axis=1) 



sb.heatmap(testdata.isnull())
testdata=testdata.fillna(method='bfill')

testdata.head()


pd.unique(traindata.Sex)
pd.unique(traindata.Embarked)
sex=pd.get_dummies(traindata['Sex'],drop_first=True)

embark=pd.get_dummies(traindata["Embarked"],drop_first=True)
traindata.drop(['PassengerId','Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
traindata1=pd.concat([traindata,sex,embark],axis=1)
X_train=traindata1.drop("Survived",axis=1)

Y_train=traindata1['Survived']
pd.unique(testdata.Sex)
pd.unique(testdata.Embarked)
sex=pd.get_dummies(testdata['Sex'],drop_first=True)

embark=pd.get_dummies(testdata["Embarked"],drop_first=True) 
testdata.drop(['PassengerId','Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
testdata=pd.concat([testdata,sex,embark],axis=1)
X_test=testdata

X_train.shape, Y_train.shape, X_test.shape
from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)

acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

acc_random_forest
random_forest.score(X_train,Y_train)
submission = pd.DataFrame({

        "PassengerId": testdat,

        "Survived": Y_pred   

})

submission.set_index('PassengerId', inplace=True) 

submission