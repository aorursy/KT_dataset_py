# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
train=pd.read_csv('../input/titanic/train.csv')

test=pd.read_csv('../input/titanic/test.csv')

datas = pd.concat([train, test],ignore_index=True)
train["FamilySize"]=train['SibSp']+train['Parch']+1

sns.barplot(x='FamilySize',y="Survived",data=train)

datas['Title'] = datas['Name'].apply(lambda x:x.split(',')[1].split('.')[0])

datas['Title'].replace([' Capt', ' Col', ' Major', ' Dr', ' Rev'],'Officer', inplace=True)

datas['Title'].replace([' Don', ' Sir', ' the Countess', ' Dona', ' Lady'], 'Royalty', inplace=True)

datas['Title'].replace([' Mme', ' Ms', ' Mrs'],'Mrs', inplace=True)

datas['Title'].replace([' Mlle', ' Miss'], 'Miss', inplace=True)

datas['Title'].replace([' Master',' Jonkheer'],'Master', inplace=True)

datas['Title'].replace([' Mr'], 'Mr', inplace=True)



sns.barplot(x="Title", y="Survived", data=datas)
datas['Fam_size']=datas['SibSp']+datas['Parch']+1

#这里逗号前是行，后面是列，相当于添加一列，此列系列=0或1或2



datas.loc[datas['Fam_size']>7,'Fam_type']=0

datas.loc[(datas['Fam_size']>=2)&(datas['Fam_size']<=4),'Fam_type']=2

datas.loc[(datas['Fam_size']>4)&(datas['Fam_size']<=7)|(datas['Fam_size']==1),'Fam_type']=1

sns.barplot(x="Fam_type", y="Survived", data=datas)
datas["Cabin"]=datas["Cabin"].fillna("U")

datas["Board"]=datas['Cabin'].str.get(0)

sns.barplot(x='Board',y='Survived',data=datas)
Ticket_counts=dict(datas['Ticket'].value_counts())

#x是变量，然后输出的可以理解为是一个数组的每个值

datas["Ticketgroup"]=datas['Ticket'].apply(lambda x:Ticket_counts[x])

sns.barplot(x='Ticketgroup',y='Survived',data=datas)
datas.loc[datas['Ticketgroup']>8,'Ticketlabels']=0



datas.loc[(datas['Ticketgroup']>4)&(datas['Ticketgroup']<=8)|(datas['Ticketgroup']==1),'Ticketlabels']=1

datas.loc[(datas['Ticketgroup']>=2)&(datas['Ticketgroup']<=4),'Ticketlabels']=2



sns.barplot(x='Ticketlabels', y='Survived', data=datas)
datas['Embarked'] = datas['Embarked'].fillna('S')
fare=datas[(datas['Embarked'] == "S") & (datas['Pclass'] == 3)].Fare.median()

datas['Fare']=datas['Fare'].fillna(fare)
from sklearn.ensemble import RandomForestRegressor

ages = datas[['Age', 'Pclass','Sex','Title']]

ages=pd.get_dummies(ages)

#转换成onehot编码，也就是将性别 title转换成数字形式

known_ages = ages[ages.Age.notnull()].values

unknown_ages = ages[ages.Age.isnull()].values

X=known_ages[:,1:]



y=known_ages[:,0]
rfr1=RandomForestRegressor(random_state=60,n_estimators=100)

rfr1.fit(X,y)

pre_ages=rfr1.predict(unknown_ages[:,1::])

datas.loc[(datas.Age.isnull()),'Age']=pre_ages
datas['Surname']=datas['Name'].apply(lambda x:x.split(',')[0].strip())

Surname_Count = dict(datas['Surname'].value_counts())

datas['FamilyGroup'] = datas['Surname'].apply(lambda x:Surname_Count[x])

Female_Child_Group=datas.loc[(datas['FamilyGroup']>=2) & ((datas['Age']<=12) | (datas['Sex']=='female'))]

Male_Adult_Group=datas.loc[(datas['FamilyGroup']>=2) & (datas['Age']>12) & (datas['Sex']=='male')]
Female_Child=pd.DataFrame(Female_Child_Group.groupby('Surname')['Survived'].mean().value_counts())

Female_Child.columns=['GroupCount11']

sns.barplot(x=Female_Child.index, y=Female_Child["GroupCount11"])
Male_adult=pd.DataFrame(Male_Adult_Group.groupby('Surname')['Survived'].mean().value_counts())

Male_adult.columns=['Groupcount']

Male_adult
Female_Child_List=Female_Child_Group.groupby('Surname')['Survived'].mean()

Dead_list=set(Female_Child_List[Female_Child_List.apply(lambda x:x==0)].index)

print(Dead_list)

Male_Adult_List=Male_Adult_Group.groupby('Surname')['Survived'].mean()

Survived_list=set(Male_Adult_List[Male_Adult_List.apply(lambda x:x==1)].index)

print(Survived_list)
test=datas.loc[datas['Survived'].isnull()]

train=datas.loc[datas['Survived'].notnull()]

test.loc[(test['Surname'].apply(lambda x:x in Dead_list)),'Sex']='male'#因为Surname是在Dead_list中存贮的变量

test.loc[(test['Surname'].apply(lambda x:x in Dead_list)),'Age'] = 60

test.loc[(test['Surname'].apply(lambda x:x in Dead_list)),'Title'] = 'Mr'

test.loc[(test['Surname'].apply(lambda x:x in Survived_list)),'Sex'] = 'female'

test.loc[(test['Surname'].apply(lambda x:x in Survived_list)),'Age'] = 5

test.loc[(test['Surname'].apply(lambda x:x in Survived_list)),'Title'] = 'Miss'
datas=pd.concat([train,test])

datas=datas[['Survived','Pclass','Sex','Age','Fare','Embarked','Title','Fam_type','Board','Ticketlabels']]

datas=pd.get_dummies(datas)

train=datas[datas['Survived'].notnull()]

test=datas[datas['Survived'].isnull()].drop('Survived',axis=1)#删除“Survived”为标签的那一列

X = train.values[:,1:]

y = train.values[:,0]
from sklearn.linear_model import LogisticRegression as LR

from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

from sklearn.feature_selection import SelectFromModel

from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestClassifier
lr = LR(penalty="l2",solver="liblinear",C=0.9850000000000004,max_iter=200).fit(X,y)

cross_val_score(lr,X,y,cv=10).mean()
predictions = lr.predict(test)

test=pd.read_csv('../input/titanic/test.csv')

PassengerId=test['PassengerId']

prdict_test = pd.DataFrame({"PassengerId": PassengerId, "Survived": predictions.astype(np.int32)})

prdict_test.to_csv("submission.csv", index=False)