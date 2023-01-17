import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from matplotlib import rcParams

import seaborn as sb

%matplotlib inline
train=pd.read_csv('../input/train.csv')
for item in [train]:

    item['FamilySize']=item['Parch']+item['SibSp']

    item['Age']=item['Age'].fillna(item['Age'].median())    

train.head()
figure, [ax1,ax2,ax3,ax4]=plt.subplots(4,1,figsize=(12,12))



sb.violinplot(x='Pclass',y='Survived',hue='Sex',data=train,split=True,inner='point',ax=ax1)

sb.violinplot(x='FamilySize',y='Survived',hue='Sex',data=train,split=True,inner='point',ax=ax2)

sb.violinplot(x='Embarked',y='Survived',hue='Sex',data=train,split=True,inner='point',ax=ax3)

sb.violinplot(x='Survived',y='Age',hue='Sex',data=train,split=True,inner='point',ax=ax4)
survivedAge=train.loc[(train['Survived']==1),'Age']

deadAge=train.loc[(train['Survived']==0),'Age'];

sb.distplot(survivedAge)

sb.distplot(deadAge,color='r')

plt.ylabel('Survival Probablity');plt.xlabel('Age');plt.ylim(0,0.08);plt.xlim(0,60)

plt.legend(['Survived','Dead'])

[survivedAge.mean(), deadAge.mean(),survivedAge.std(),deadAge.std()]
for item in [train]:    

    item['Adult']=item['Age'].apply(lambda x:0 if x<10 else 1)

    item['Male']=item['Sex'].apply(lambda x:0 if x=='female' else 1)

    item['EmbarkedNum']=item['Embarked'].apply(lambda x:0 if x=='S' else (1 if x=='C' else 2 ))

    item['Class']=item['Pclass'].apply(lambda x:1 if (x==1 or x==2) else 0)

train=train.drop(['PassengerId','Pclass','Sex','Name','SibSp','Parch','Ticket','Fare','Cabin','Embarked','Age'],axis=1)

train.head()
import sklearn

from sklearn.linear_model import LogisticRegression

lg=LogisticRegression().fit(train[['Male','Class','EmbarkedNum','FamilySize']],train['Survived'])

print(lg.score(train[['Male','Class','EmbarkedNum','FamilySize']],train['Survived']))
test=pd.read_csv('../input/test.csv')

for item in [test]:

    item['FamilySize']=item['Parch']+item['SibSp']

    item['Age']=item['Age'].fillna(item['Age'].median())  

    item['Adult']=item['Age'].apply(lambda x:0 if x<10 else 1)

    item['Male']=item['Sex'].apply(lambda x:0 if x=='female' else 1)

    item['EmbarkedNum']=item['Embarked'].apply(lambda x:0 if x=='S' else (1 if x=='C' else 2 ))

    item['Class']=item['Pclass'].apply(lambda x:1 if (x==1 or x==2) else 0)

test=test.drop(['Pclass','Sex','Name','SibSp','Parch','Ticket','Fare','Cabin','Embarked','Age'],axis=1)
Answer=pd.DataFrame()

Answer['PassengerID']=test['PassengerId']

Answer['Survived']=lg.predict(test[['Male','Class','EmbarkedNum','FamilySize']])
Answer.to_csv('Answer.csv',index=False)