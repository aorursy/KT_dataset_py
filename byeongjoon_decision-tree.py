import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

train=pd.read_csv("/kaggle/input/titanic/train.csv")

test=pd.read_csv("/kaggle/input/titanic/test.csv")
train['Sex'].loc[train['Sex']=='male']=0

train['Sex'].loc[train['Sex']=='female']=1



test['Sex'].loc[test['Sex']=='male']=0

test['Sex'].loc[test['Sex']=='female']=1
test.loc[test['Fare'].isnull()]
test["Fare"].loc[test['Fare'].isnull()]=8.0500
train['Cabin'].loc[train['Cabin'].isnull()]=0

train['Cabin'].loc[train['Cabin']!=0]=1



test['Cabin'].loc[test['Cabin'].isnull()]=0

test['Cabin'].loc[test['Cabin']!=0]=1
train=train.dropna(subset=['Embarked'])
train['Embarked'].loc[train['Embarked']=='S']=1

train['Embarked'].loc[train['Embarked']=='C']=2

train['Embarked'].loc[train['Embarked']=='Q']=3



test['Embarked'].loc[test['Embarked']=='S']=1

test['Embarked'].loc[test['Embarked']=='C']=2

test['Embarked'].loc[test['Embarked']=='Q']=3
df=pd.concat([train,test])
print(df['Age'].loc[(df['Sex']==0)].mean())

print(df['Age'].loc[(df['Sex']==1)].mean())
def hist(par):

    plt.figure(figsize=(20,6));

    plt.hist(df[par].loc[df['Survived']==1],bins=20,alpha=0.6,label='live');

    plt.hist(df[par].loc[df['Survived']==0],bins=20,alpha=0.6,label='die');

    plt.grid()

    plt.legend()

    plt.show()
hist('Age')
train['Age'].loc[(train['Age'].isnull())&(train['Sex']==0)]=df['Age'].loc[(df['Sex']==0)].mean()

train['Age'].loc[(train['Age'].isnull())&(train['Sex']==1)]=df['Age'].loc[(df['Sex']==1)].mean()



test['Age'].loc[(test['Age'].isnull())&(test['Sex']==0)]=df['Age'].loc[(df['Sex']==0)].mean()

test['Age'].loc[(test['Age'].isnull())&(test['Sex']==1)]=df['Age'].loc[(df['Sex']==1)].mean()

train['Age'].loc[train['Age']<=10]=0

train['Age'].loc[(10<train['Age'])&(train['Age']<=15)]=1

train['Age'].loc[(15<train['Age'])&(train['Age']<=30)]=2

train['Age'].loc[(30<train['Age'])&(train['Age']<=50)]=3

train['Age'].loc[(50<train['Age'])]=4



test['Age'].loc[test['Age']<=10]=0

test['Age'].loc[(10<test['Age'])&(test['Age']<=15)]=1

test['Age'].loc[(15<test['Age'])&(test['Age']<=30)]=2

test['Age'].loc[(30<test['Age'])&(test['Age']<=50)]=3

test['Age'].loc[(50<test['Age'])]=6
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(max_depth=6,random_state=1)
# x=['Pclass', 'Sex','SibSp','Parch','Fare','Cabin','Embarked','Age']

x=['Pclass', 'Sex','SibSp','Parch','Fare','Cabin','Embarked','Age']

clf = clf.fit(train[x],

              train['Survived'])

from sklearn.model_selection import cross_val_score

scores = cross_val_score(clf, train[x], train['Survived'],cv=10)

scores.mean()
predict=clf.predict(test[x])
sub=test[['PassengerId']]

# sub=pd.DataFrame(test[['PassengerId']],predict)
sub['Survived']=predict[:]
sub.to_csv("submission.csv",index=False)