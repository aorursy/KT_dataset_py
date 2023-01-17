# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from matplotlib import style

style.use('dark_background')

from numpy import nan

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data_dir='/kaggle/input/titanic/'
train=pd.read_csv(data_dir+'train.csv')

test=pd.read_csv(data_dir+'test.csv')

fig,axes=plt.subplots(nrows=1,ncols=2,figsize=(15,8))

women=train[train['Sex']=='female']

men=train[train['Sex']=='male']

ax=sns.distplot(women[women['Survived']==1].Age.dropna(),bins=18,label='Survived',ax=axes[0],kde=False)

ax=sns.distplot(women[women['Survived']==0].Age.dropna(),bins=18,label='Not_Survived',ax=axes[0],kde=False)

ax.legend()

ax.set_title('Female')

ax=sns.distplot(men[men['Survived']==1].Age.dropna(),bins=18,label='Survived',ax=axes[1],kde=False)

ax=sns.distplot(men[men['Survived']==0].Age.dropna(),bins=18,label='Not_Survived',ax=axes[1],kde=False)

ax.legend()

ax.set_title('Male')
train
test
otest=test.copy()
train.drop(['PassengerId'],axis=1,inplace=True)

test.drop(['PassengerId'],axis=1,inplace=True)
from sklearn.preprocessing import LabelEncoder

train['Sex']=LabelEncoder().fit_transform(train['Sex'])

test['Sex']=LabelEncoder().fit_transform(test['Sex'])
train['Name']=train['Name'].map(lambda x: x.split(',')[1].split('.')[0].strip())

test['Name']=test['Name'].map(lambda x: x.split(',')[1].split('.')[0].strip())
train
test
titles=train['Name'].unique()

titles
test['Name'].unique()
test[test['Name']=='Ms']
train.isnull().sum()
test.isnull().sum()
train['Age'].fillna(-1,inplace=True)



medians={}

for title in titles:

    med=train.Age[(train['Age']!=-1) & (train['Name']==title)].median()

    medians[title]=med



for index,rows in train.iterrows():

    if rows['Age']==-1:

        train.loc[index,'Age']=medians[rows['Name']]



train        
medians
train.isnull().sum()
test.isnull().sum()
titles1=test['Name'].unique()

titles1
test['Age'].fillna(-1,inplace=True)



medians1={}

for title in titles1:

    med1=test.Age[(test['Age']!=-1) & (test['Name']==title)].median()

    if title=='Ms':

        medians1[title]=28.0

    else:

        medians1[title]=med1



for index,rows in test.iterrows():

    if rows['Age']==-1:

        test.loc[index,'Age']=medians1[rows['Name']]



test     
from sklearn.preprocessing import StandardScaler

train['Age']=StandardScaler().fit_transform(train['Age'].values.reshape(-1,1))

test['Age']=StandardScaler().fit_transform(test['Age'].values.reshape(-1,1))
test.isnull().sum()
train.isnull().sum()
fig=plt.figure(figsize=(20,10))

i=1

for title in train['Name'].unique():

    fig.add_subplot(3,6,i)

    plt.title('Title :{}'.format(title))

    train.Survived[train['Name']==title].value_counts().plot(kind='pie')

    i+=1
titles
test
subs={

    'Capt':0,

    'Jonkheer':0,

    'Rev':0,

    'Don':0,

    'Mr':1,

    'Major':2,

    'Col':2,

    'Dr':3,

    'Master':3,

    'Miss':4,

    'Mrs':5,

    'Mme': 6,

    'Ms': 6,

    'Mlle': 6,

    'Sir': 6,

    'Lady': 6,

    'the Countess': 6

    

    

}

train['Name']=train['Name'].map(subs)

from sklearn.preprocessing import StandardScaler

train['Name']=StandardScaler().fit_transform(train['Name'].values.reshape(-1,1))
subs1={

  

   'Rev':0,

    'Dona':4,           #'Mr', 'Mrs', 'Miss', 'Master', 'Ms', 'Col', 'Rev', 'Dr', 'Dona'

   'Mr':1,

    

    'Col':2,

   'Dr':3,

   'Master':3,

    'Miss':4,

   'Mrs':5,

  

   'Ms': 6

 

    

    

}

test['Name']=test['Name'].map(subs1)

test['Name']=StandardScaler().fit_transform(test['Name'].values.reshape(-1,1))
train.isnull().sum()
test.isnull().sum()
test['Fare'].fillna(-1,inplace=True)

medians2={}

for pc in test['Pclass'].unique():

    med2=test.Fare[(test['Fare']!=-1) & (test['Pclass']==pc)].median()

    medians2[pc]=med2

    

for index,row in test.iterrows():

    if row['Fare']==-1:

        test.loc[index,'Fare']=medians2[row['Pclass']]

test.isnull().sum()
train.isnull().sum()
train['Fare']=StandardScaler().fit_transform(train['Fare'].values.reshape(-1,1))

test['Fare']=StandardScaler().fit_transform(test['Fare'].values.reshape(-1,1))

fig=plt.figure(figsize=(20,10))

i=1

for pc in train['Pclass'].unique():

    fig.add_subplot(1,3,i)

    plt.title('class : {}'.format(pc))

    train.Survived[train['Pclass']==pc].value_counts().plot(kind='pie')

    i=i+1
train['Pclass']=StandardScaler().fit_transform(train['Pclass'].values.reshape(-1,1))

test['Pclass']=StandardScaler().fit_transform(test['Pclass'].values.reshape(-1,1))

train
test
fig=plt.figure(figsize=(20,10))

i=1

for parch in train['Parch'].unique():

    fig.add_subplot(3,4,i)

    plt.title('parent child {}'.format(parch))

    train.Survived[train['Parch']==parch].value_counts().plot(kind='pie')

    i=i+1
test['Parch'].unique()
train['Parch'].unique()
subs3={

   6:0, 

   4:0,

   5:1,

   0:2,

    2:3,

    1:4,

    3:5

}

train['Parch']=train['Parch'].map(subs3)

subs4={

   6:0, 

   4:0,

   9:0, 

   5:1,

   0:2,

    2:3,

    1:4,

    3:5

}

test['Parch']=test['Parch'].map(subs4)
train['Parch']=StandardScaler().fit_transform(train['Parch'].values.reshape(-1,1))

test['Parch']=StandardScaler().fit_transform(test['Parch'].values.reshape(-1,1))

test.isnull().sum()

train.isnull().sum()
train.drop(['Ticket'],axis=1,inplace=True)

test.drop(['Ticket'],axis=1,inplace=True)
train['Embarked'].value_counts()
train['Embarked'].fillna('S',inplace=True)

fig=plt.figure(figsize=(20,10))

i=1

for em in train['Embarked'].unique():

    fig.add_subplot(3,4,i)

    plt.title('Embarked {}'.format(em))

    train.Survived[train['Embarked']==em].value_counts().plot(kind='pie')

    i=i+1

    
subs5={

    

    'S':0,

    'Q':1,

    'C':2

}

train['Embarked']=train['Embarked'].map(subs5)

test['Embarked']=test['Embarked'].map(subs5)
train['Embarked']=StandardScaler().fit_transform(train['Embarked'].values.reshape(-1,1))

test['Embarked']=StandardScaler().fit_transform(test['Embarked'].values.reshape(-1,1))

fig=plt.figure(figsize=(20,10))

i=1

for sb in train['SibSp'].unique():

    fig.add_subplot(3,4,i)

    plt.title('Sibling Spouse {}'.format(sb))

    train.Survived[train['SibSp']==sb].value_counts().plot(kind='pie')

    i=i+1
subs6={

    5:0,

    8:0,

    4:1,

    3:2,

    0:3,

    2:4,

    1:5

}

train['SibSp']=train['SibSp'].map(subs6)

test['SibSp']=test['SibSp'].map(subs6)
train['SibSp']=StandardScaler().fit_transform(train['SibSp'].values.reshape(-1,1))

test['SibSp']=StandardScaler().fit_transform(test['SibSp'].values.reshape(-1,1))

train['Cabin'].fillna('U',inplace=True)

test['Cabin'].fillna('U',inplace=True)

train['Cabin']=train['Cabin'].map(lambda x:x[0])

test['Cabin']=test['Cabin'].map(lambda x:x[0])
test.isnull().sum()
train.isnull().sum()
fig=plt.figure(figsize=(20,10))

i=1

for cab in train['Cabin'].unique():

    fig.add_subplot(3,4,i)

    plt.title('Cabin {}'.format(cab))

    train.Survived[train['Cabin']==cab].value_counts().plot(kind='pie')

    i=i+1
train['Cabin'].unique()
test['Cabin'].unique()
sub7 = {

    'T': 0,

    'U': 1,

    'A': 2,

    'G': 3,

    'C': 4,

    'F': 5,

    'B': 6,

    'E': 7,

    'D': 8

}

train['Cabin']=train['Cabin'].map(sub7)
sub8= {

    'U': 1,

    'A': 2,

    'G': 3,

    'C': 4,

    'F': 5,

    'B': 6,

    'E': 7,

    'D': 8

}

test['Cabin']=test['Cabin'].map(sub8)
train['Cabin']=StandardScaler().fit_transform(train['Cabin'].values.reshape(-1,1))

test['Cabin']=StandardScaler().fit_transform(test['Cabin'].values.reshape(-1,1))
test
y=train['Survived']

train.drop(['Survived'],axis=1,inplace=True)
train
from sklearn.ensemble import RandomForestClassifier

rf=RandomForestClassifier(n_estimators=100)

rf.fit(train,y)

y_pred=rf.predict(test)




# random_forest.fit(x_train,y_train)

# y_pred=random_forest.predict(x_test)

# acc_random_forest=np.round(random_forest.score(x_train,y_train)*100,2)

# print(f' The accuracy of random forest is {acc_random_forest}')



from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score

ml=list(range(150))

lst_n=[i for i in ml if i!=0]

cv_scores=[]

MSE=[]

for k in lst_n:

    #print(k,end='')

    random_forest=RandomForestClassifier(n_estimators=k)

    scores=cross_val_score(random_forest,train,y,cv=10,scoring='accuracy')

    cv_scores.append(scores.mean())

    

MSE=[1-x for x in cv_scores]

best_n=lst_n[MSE.index(min(MSE))]

print(f'best n of Random Forest is {best_n}')

plt.plot(lst_n,MSE)

plt.xlabel('n value of Random Forest ->')

plt.ylabel('Mean Squared Error')

plt.show()

rf1=RandomForestClassifier(n_estimators=best_n)

rf1.fit(train,y)

y_pred_rf=rf1.predict(test)

otest.columns
sub=pd.DataFrame()

sub['PassengerId']=otest['PassengerId']

sub['Survived']=y_pred_rf



sub
len(sub)
sub.to_csv('randomforest.csv',index=False)

print('submission is ready')