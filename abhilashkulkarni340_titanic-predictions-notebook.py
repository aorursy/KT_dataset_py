import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn import preprocessing
trainds=pd.read_csv('../input/train.csv')
trainds.head()
trainds.isnull().any()
trainds2=trainds
trainds2['Age'].fillna(trainds2['Age'].mean(axis=0),inplace=True)
trainds2['Embarked'].fillna('S',inplace=True)
trainds2.corr()
trainds2.groupby(['Embarked']).count()
trainds2['Sex'][trainds2['Sex']=='male']=0
trainds2['Sex'][trainds2['Sex']=='female']=1
trainds2['Age'][trainds2['Age']<18]=1
trainds2['Age'][trainds2['Age']>17]=0
trainds2['Embarked'][trainds2['Embarked']=='S']=0
trainds2['Embarked'][trainds2['Embarked']=='C']=1
trainds2['Embarked'][trainds2['Embarked']=='Q']=2
trainds2.head()
min_max=preprocessing.MinMaxScaler()
x_train_main=min_max.fit_transform(trainds2[['Age','Sex','Fare']])
x_train_main
trainds2.head()
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(trainds2[['Age','Sex','Fare']],trainds2['Survived'],test_size=0.25)
x_train_minmax=min_max.fit_transform(x_train)
x_test_minmax=min_max.fit_transform(x_test)
print(x_train_minmax)
print(x_test_minmax)
rfc=RandomForestClassifier(n_estimators=10)
rfc.fit(x_train_minmax,y_train)
accuracy_score(y_test,rfc.predict(x_test_minmax))
trainds2_minmax=min_max.fit_transform(trainds2[['Age','Sex','Fare']])
testds=pd.read_csv('../input/test.csv')
testds2=testds
testds2['Age'].fillna(testds2['Age'].mean(axis=0),inplace=True)
testds2['Embarked'].fillna('S',inplace=True)
testds2['Fare'].fillna(testds2['Fare'].mean(axis=0),inplace=True)
testds2.isnull().any()
testds2['Sex'][testds2['Sex']=='male']=0
testds2['Sex'][testds2['Sex']=='female']=1
testds2['Age'][testds2['Age']<18]=1
testds2['Age'][testds2['Age']>17]=0
testds2['Embarked'][testds2['Embarked']=='S']=0
testds2['Embarked'][testds2['Embarked']=='C']=1
testds2['Embarked'][testds2['Embarked']=='Q']=2
testds_minmax=min_max.fit_transform(testds2[['Age','Sex','Fare']])
gen=pd.read_csv('../input/gender_submission.csv')
gen.head()
rfc.fit(trainds2_minmax,trainds2['Survived'])
accuracy_score(gen['Survived'],rfc.predict(testds_minmax))

