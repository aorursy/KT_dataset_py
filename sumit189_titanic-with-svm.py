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
import pandas as pd
train=pd.read_csv('/kaggle/input/titanic/train.csv')

test=pd.read_csv('/kaggle/input/titanic/test.csv')
train.head()
test.head()
train.shape
test.shape
train.info()
train.isnull().sum()
import matplotlib.pyplot as plt

import seaborn as sns

sns.set()
def bar_chart(feature):

  survived=train[train['Survived']==1][feature].value_counts()

  dead=train[train['Survived']==0][feature].value_counts()

  df=pd.DataFrame([survived,dead])

  df.index=['Survived','Dead']

  df.plot(kind='bar',stacked=True,figsize=(10,5))
bar_chart('Sex')
bar_chart('Pclass')
bar_chart('SibSp')
bar_chart('Parch')
bar_chart('Embarked')
#Extract Titles from name

dataset=[train,test]



for d in dataset:

  d['Title']=d['Name'].str.extract(' ([A-Za-z]+)\.',expand=False)
train['Title'].value_counts()
mapping={"Mr":0,"Miss":1,"Mrs":2,"Master":3,"Dr":3,"Rev":3,"Col":3,"Major":3,"Mlle":3,"Sir":3,"Capt":3,"Lady":3,

         "Mme":3,"Countess":3,"Jonkheer":3,"Don":3,"Ms":3}
for d in dataset:

  d.Title=d.Title.map(mapping)
test['Name'].iloc[414]
test['Age'].iloc[414]
test['Parch'].iloc[414]
test.Title.iloc[414]=2.0
test['Title'].iloc[414]
bar_chart('Title')
train.drop('Name',axis=1,inplace=True)

test.drop('Name',axis=1,inplace=True)

train.head()
test.head()
#mapping genders

sex_mapping={"male":0,"female":1}

for d in dataset:

  d.Sex=d.Sex.map(sex_mapping)
train.head()
#age filling na values

train.Age.fillna(train.groupby("Title")['Age'].transform("median"),inplace=True)

test.Age.fillna(test.groupby("Title")['Age'].transform("median"),inplace=True)
facet=sns.FacetGrid(train,hue="Survived",aspect=4)

facet.map(sns.kdeplot,'Age',shade=True)

facet.set(xlim=(0,train['Age'].max()))

facet.add_legend()

plt.show()
facet=sns.FacetGrid(train,hue="Survived",aspect=4)

facet.map(sns.kdeplot,'Age',shade=True)

facet.set(xlim=(0,train['Age'].max()))

facet.add_legend()

plt.xlim(0,20)
facet=sns.FacetGrid(train,hue="Survived",aspect=4)

facet.map(sns.kdeplot,'Age',shade=True)

facet.set(xlim=(0,train['Age'].max()))

facet.add_legend()

plt.xlim(20,30)
facet=sns.FacetGrid(train,hue="Survived",aspect=4)

facet.map(sns.kdeplot,'Age',shade=True)

facet.set(xlim=(0,train['Age'].max()))

facet.add_legend()

plt.xlim(30,40)
facet=sns.FacetGrid(train,hue="Survived",aspect=4)

facet.map(sns.kdeplot,'Age',shade=True)

facet.set(xlim=(0,train['Age'].max()))

facet.add_legend()

plt.xlim(40,60)
for d in dataset:

  d.loc[d.Age<=16,'Age']=0,

  d.loc[(d.Age>16) & (d.Age<=26),'Age']=1,

  d.loc[(d.Age>26) & (d.Age<=36),'Age']=2,

  d.loc[(d.Age>36) & (d.Age<=62),'Age']=3,

  d.loc[d.Age>62,'Age']=4



Pclass1=train[train['Pclass']==1]['Embarked'].value_counts()

Pclass2=train[train['Pclass']==2]['Embarked'].value_counts()

Pclass3=train[train['Pclass']==3]['Embarked'].value_counts()

df=pd.DataFrame([Pclass1,Pclass2,Pclass3])

df.index=['1st Class','2nd Class','3rd Class']

df.plot(kind='bar',stacked=True,figsize=(10,5))
for d in dataset:

  d.Embarked=d.Embarked.fillna('S')
train.head()
Embarked_mapping={"S":0,"C":1,"Q":1}
for d in dataset:

  d.Embarked=d.Embarked.map(Embarked_mapping)
train.info()
train.Fare.fillna(train.groupby('Pclass')['Fare'].transform('median'),inplace=True)

test.Fare.fillna(test.groupby('Pclass')['Fare'].transform('median'),inplace=True)
facet=sns.FacetGrid(train,hue="Survived",aspect=4)

facet.map(sns.kdeplot,'Fare',shade=True)

facet.set(xlim=(0,train['Fare'].max()))

facet.add_legend()



plt.show()
facet=sns.FacetGrid(train,hue="Survived",aspect=4)

facet.map(sns.kdeplot,'Fare',shade=True)

facet.set(xlim=(0,train['Fare'].max()))

facet.add_legend()



plt.xlim(0,30)
facet=sns.FacetGrid(train,hue="Survived",aspect=4)

facet.map(sns.kdeplot,'Fare',shade=True)

facet.set(xlim=(0,train['Fare'].max()))

facet.add_legend()



plt.xlim(0)
for d in dataset:

  d.loc[d.Fare<=17,'Fare']=0,

  d.loc[(d.Fare>17) & (d.Fare<=30),'Fare']=1,

  d.loc[(d.Fare>30) & (d.Fare<=100),'Fare']=2,

  d.loc[d.Fare>100,'Fare']=4
train.head()
#cabin

for d in dataset:

  d.Cabin=d.Cabin.str[:1]
Pclass1=train[train['Pclass']==1]['Cabin'].value_counts()

Pclass2=train[train['Pclass']==2]['Cabin'].value_counts()

Pclass3=train[train['Pclass']==3]['Cabin'].value_counts()

df=pd.DataFrame([Pclass1,Pclass2,Pclass3])

df.index=['1st class','2nd class','3rd class']

df.plot(kind='bar',stacked=True,figsize=(10,5))
cabbin_mapping={"A":0,"B":0.4,"C":0.8,"D":1.2,"E":1.6,"F":2.0,"G":2.4,"T":2.8}

for d in dataset:

  d.Cabin=d.Cabin.map(cabbin_mapping)
train.Cabin.fillna(train.groupby("Pclass")['Cabin'].transform('median'),inplace=True)

test.Cabin.fillna(test.groupby("Pclass")['Cabin'].transform('median'),inplace=True)
train['FamilySize']=train['SibSp']+train['Parch']+1

test['FamilySize']=test['SibSp']+test['Parch']+1
facet=sns.FacetGrid(train,hue="Survived",aspect=4)

facet.map(sns.kdeplot,'FamilySize',shade=True)

facet.set(xlim=(0,train['FamilySize'].max()))

facet.add_legend()

plt.xlim(0)
fam_map={1:0,2:0.4,3:0.8,4:1.2,5:1.6,6:2,7:2.4,8:2.8,9:3.2,10:3.6,11:4}

fam_map
for d in dataset:

  d.FamilySize=d.FamilySize.map(fam_map)
features_to_drop=['Ticket','SibSp','Parch']

train=train.drop(features_to_drop,1)

test=test.drop(features_to_drop,1)

train=train.drop(['PassengerId'],1)
target=train['Survived']

train=train.drop(['Survived'],1)
train.head()
from sklearn.model_selection import KFold,cross_val_score

from sklearn.svm import SVC

clf=SVC(kernel='rbf')

k_fold=KFold(n_splits=10,shuffle=True,random_state=0)

scoring="accuracy"

score=cross_val_score(clf,train,target,cv=k_fold,n_jobs=1,scoring=scoring)

print(score)
import numpy as np

round(np.mean(score)*100,2)
test_data=test.drop('PassengerId',axis=1).copy()
clf=SVC()

clf.fit(train,target)

prediction=clf.predict(test_data)
prediction
submission=pd.DataFrame({

    "PassengerId":test['PassengerId'],

    "Survived":prediction

})

submission.to_csv('submission.csv',index=False)