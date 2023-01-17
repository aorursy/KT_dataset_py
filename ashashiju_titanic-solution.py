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
train=pd.read_csv("../input/titanic/train.csv")

test=pd.read_csv("../input/titanic/test.csv")

train.head()
test.info()
train.info()

import matplotlib as plt

%matplotlib inline 

import seaborn as sns

sns.set()
def barchart(feature):

    survived=train[train['Survived']==1][feature].value_counts()

    dead=train[train['Survived']==0][feature].value_counts()

    df=pd.DataFrame([survived,dead])

    df.index=['Survived','Dead']

    df.plot(kind='bar',stacked=True,figsize=(5,5))







barchart('Sex')
barchart('Pclass')
barchart('SibSp')
barchart('Embarked')
trainTestData=[train,test]

for ds in trainTestData:

    ds['Title']=ds['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
train['Title'].value_counts()
test['Title'].value_counts()
titleMapping={'Mr':0,'Miss':1,'Mrs':2,'Master':3,'Dr':3,'Don':3,'Dona':3,'Mme':3,'Countess':3,'Rev':3,'Sir':3,'Lady':3,

              'Capt':3,'Major':3,'Mlle':3,'Jonkheer':3,'Ms':3,'Col':3}
for ds in trainTestData :

    ds['Title']=ds['Title'].map(titleMapping)

train.head()
test.head()
train.drop('Name',axis=1,inplace=True)

test.drop('Name',axis=1,inplace=True)
test.head()
sexMapping={'male':0,'female':1}

for ds in trainTestData :

    ds['Sex']=ds['Sex'].map(sexMapping)

train.head()
train.info()

train['Age'].fillna(train.groupby('Title')['Age'].transform('median'),inplace=True)

test['Age'].fillna(test.groupby('Title')['Age'].transform('median'),inplace=True)
for ds in trainTestData:

    ds.loc[ds['Age']<=16,'Age']=0

    ds.loc[(ds['Age']>16) & (ds['Age']<=26),'Age']=1,

    ds.loc[(ds['Age']>26) & (ds['Age']<=36),'Age']=2,

    ds.loc[(ds['Age']>36) & (ds['Age']<=62),'Age']=3,

    ds.loc[ds['Age']>62,'Age']=4

           
test.info()
for ds in trainTestData:

    ds['Embarked']=ds['Embarked'].fillna('S')

train.head()

embarked_mapping = {"S": 0, "C": 1, "Q": 2}

for dataset in trainTestData:

    dataset['Embarked'] = dataset['Embarked'].map(embarked_mapping)
train.info()
train['Fare'].fillna(train.groupby('Pclass')['Fare'].transform('median'),inplace=True)

test['Fare'].fillna(test.groupby('Pclass')['Fare'].transform('median'),inplace=True)
#facet=sns.FacetGrid(train,hue='Survived',aspect=4)

#facet.map(sns.kdeplot,'Fare',shade=True)

#facet.set(xlim=(0,train['Fare'].max()))

#facet.add_legend()

#plt.show()

for ds in trainTestData:

    ds.loc[ds['Fare']<=17,'Fare']=0

    ds.loc[(ds['Fare']>17) & (ds['Fare']<=30),'Fare']=1,

    ds.loc[(ds['Fare']>30) & (ds['Fare']<=100),'Fare']=2,

    ds.loc[ds['Fare']>100,'Fare']=4
train.head()
for ds in trainTestData:

    ds['Cabin']=ds['Cabin'].str[:1]
#Pcls1=train[train['Pclass']==1]['Cabin'].value_counts()

#Pcls2=train[train['Pclass']==2]['Cabin'].value_counts()

#Pcls3=train[train['Pclass']==3]['Cabin'].value_counts()

#df=pd.DataFrame([Pcls1,Pcls2,Pcls3])



#df.index=['Ist Class','2nd Class','3rd Class']

#df.plot(kind='bar',stacked=True,figsize=(10,5))

cabinMapping={'A':0,'B':0.4,'C':0.8,'D':1.2,'E':1.6,'F':2,'G':3,'Mme':2.4,'T':2.8}

for ds in trainTestData :

    ds['Cabin']=ds['Cabin'].map(cabinMapping)

train.head()
train['Cabin'].fillna(train.groupby('Pclass')['Cabin'].transform('median'),inplace=True)

test['Cabin'].fillna(test.groupby('Pclass')['Cabin'].transform('median'),inplace=True)
train['FamilySize']=train['SibSp']+train['Parch']+1

test['FamilySize']=test['SibSp']+test['Parch']+1
familyMapping={1:0,2:0.4,3:0.8,4:1.2,5:1.6,6:2,7:2.4,8:2.8,9:3.2,10:3.6,11:4 }

for ds in trainTestData :

    ds['FamilySize']=ds['FamilySize'].map(familyMapping)

train.head()
train.head()
features_drop = ['Ticket', 'SibSp', 'Parch']

train = train.drop(features_drop, axis=1)

test = test.drop(features_drop, axis=1)

train = train.drop(['PassengerId'], axis=1)

test.shape
train_data = train.drop('Survived', axis=1)

target = train['Survived']



train_data.shape, target.shape

train_data.head(10)
from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC



import numpy as np
train.info()
from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score



clf = SVC()

clf.fit(train_data, target)



test_data = test.drop("PassengerId", axis=1).copy()

prediction = clf.predict(test_data)
print(prediction)

submission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": prediction

    })



submission.to_csv('submission.csv', index=False)
submission = pd.read_csv('submission.csv')

submission.head()