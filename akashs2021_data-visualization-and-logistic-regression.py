import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

data =pd.read_csv('../input/train.csv')
data.head()
sns.countplot(x="Survived",data=data)
sns.countplot(x="Survived",hue="Sex",data=data)
sns.countplot(x="Survived",hue="Pclass",data=data)
data["Age"].plot.hist()
data.isnull()[:10] ## Returns true if the value is NULL otherwise false
data.isnull().sum() 
data.drop('Cabin',axis=1,inplace=True)
data.dropna(inplace=True)#drop every row which has null value
data.head()
sex=pd.get_dummies(data['Sex'],drop_first=True)

sex.head()
embark=pd.get_dummies(data['Embarked'],drop_first=True)

embark.head()
pcl=pd.get_dummies(data['Pclass'],drop_first=True)

pcl.head()
data=pd.concat([data,sex,pcl,embark],axis=1)

data.head()
data=data.drop(['Pclass','Sex','Embarked','Name','Ticket','PassengerId'],axis=1)

data.head()
x=data.drop('Survived',axis=1)

y=data['Survived']
print(x.head())

print(y.head())
import sklearn

from sklearn.linear_model import LogisticRegression

logmodel=LogisticRegression()

logmodel.fit(x,y)
test=pd.read_csv('../input/test.csv')
test.head()
passengerid=test['PassengerId'].values

passengerid[:10]
sex=pd.get_dummies(test['Sex'],drop_first=True)

sex.head()
embark=pd.get_dummies(test['Embarked'],drop_first=True)

embark.head()
pcl=pd.get_dummies(test['Pclass'],drop_first=True)

pcl.head()
test=pd.concat([test,sex,pcl,embark],axis=1)

test.head()
test=test.drop(['Pclass','Sex','Embarked','Name','Ticket','PassengerId','Cabin'],axis=1)

test.head()
test.fillna(value=0,inplace=True)#fill values

test.head()
predictions=logmodel.predict(test)
predictions[:10]
predictions=predictions.tolist()

predictions[:10]
passengerid=passengerid.tolist()

passengerid[:10]
len(predictions)
submission={

    'PassengerId':passengerid,

    'Survived':predictions

}


df = pd.DataFrame(submission)
df.head()
df.to_csv('submission.csv',index=False)