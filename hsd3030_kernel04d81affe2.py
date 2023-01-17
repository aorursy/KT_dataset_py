# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.



train = pd.read_csv('../input/train.csv')



test = pd.read_csv('../input/test.csv')



train.head()



train.describe()



train.info()



train.isnull().sum()



import matplotlib as plt

import seaborn as sns



sns.set()





feature = 'Sex'

surviver = train[train['Survived']==0][feature].value_counts()

dead = train[train['Survived']==1][feature].value_counts()

df = pd.DataFrame([surviver,dead])

df.index=['Surviever','Dead']







def chart_bar(feature):

    surviver = train[train['Survived']==0][feature].value_counts()

    dead = train[train['Survived']==1][feature].value_counts()

    df = pd.DataFrame([surviver,dead])

    df.index=['Surviever','Dead']

    df.plot(kind='bar',stacked=True,figsize=(10,5))





chart_bar('Sex')



# 동반한 형재자매배우자

chart_bar('SibSp')



chart_bar('Pclass')











# 부모와 자녀수에 따른 여부

chart_bar('Parch')



train.describe(include='all')



train.head()



chart_bar('Embarked')



train = train.drop(['Cabin'],axis=1)

test = test.drop(['Cabin'],axis=1)



train = train.drop(['Ticket'],axis=1)

test = test.drop(['Ticket'],axis=1)





# train = train.drop(['Name'],axis=1)

# test = test.drop(['Name'],axis=1)



train.describe(include='all')

train[train['Embarked']== 'S'].shape[0]

train[train['Embarked']=='C'].shape[0]

train[train['Embarked']=='Q'].shape[0]
train = train.fillna({'Embarked':'S'})



embar_map={'S':1,'C':2,'Q':3}



train['Embarked']=train['Embarked'].map(embar_map)



test = test.fillna({'Embarked':'S'})



test['Embarked']=test['Embarked'].map(embar_map)
train.Embarked.values[600].__class__
train.head()







train['Title'] = train.Name.str.extract(' ([A-Za-z]+)\.',expand=False)

test['Title'] = test.Name.str.extract(' ([A-Za-z]+)\.',expand=False)

pd.crosstab(train['Title'],train['Sex'])
title_mapping={'Mr':1,"Miss":2,"Mrs":3,"Master":4,"Royal":5,"Rare":6}



train['Title'] = train['Title'].map(title_mapping)

train['Title'] = train['Title'].fillna(0)



test['Title'] = test['Title'].map(title_mapping)

test['Title'] = test['Title'].fillna(0)

    

train.head()



train = train.drop(['Name','PassengerId'],axis=1)





test = test.drop(['Name','PassengerId'],axis =1)



train.head()



sex_mapping = {'male':0,'female':1}



train['Sex'] = train['Sex'].map(sex_mapping)

test['Sex'] = test['Sex'].map(sex_mapping)

train['Age']=train['Age'].fillna(-1)

print(len(train['Age']))

print(len(train[train['Age']==-1]))
\
# 나이를 구하기 위해서 

for index, item in train[train['Age']==-1].iterrows():

    tt = train[(train['Parch'] == item['Parch']) & (train['SibSp']==item['SibSp'])]

    kk = train[(train['Parch'] == item['Parch'])]

    pp = train[(train['SibSp'] == item['SibSp'])]

    tt = tt[tt.Age != -1]

    kk = kk[kk.Age != -1]

    pp = pp[pp.Age != -1]

#     print(tt,"\n")

    if not tt.empty:

        train.at[index,'Age'] = tt['Age'].mean()

#         train.ix[index] = tt['Age'].mean()

    elif not kk.empty:

        train.at[index,'Age'] = kk['Age'].mean()

#         train.ix[index] = kk['Age'].mean()

    elif not pp.empty:

#         train.ix[index] = pp['Age'].mean()

        train.at[index,'Age'] = pp['Age'].mean()

    else :

        train.iloc[index].Age = train['Age'].mean()

#         같이 승선한 사람이 없는 경우! Parch 와 SibSp모두 0인 경우!

for index,item in test[test['Age']==-1].iterrows():

    tt = test[(test['Parch'] == item['Parch']) & (test['SibSp']==item['SibSp'])]

    kk = test[(test['Parch'] == item['Parch'])]

    pp = test[(test['SibSp'] == item['SibSp'])]

    tt = tt[tt.Age != -1]

    kk = kk[kk.Age != -1]

    pp = pp[pp.Age != -1]

    if not tt.empty:

        test.at[index,'Age'] = tt['Age'].mean()

#         train.ix[index] = tt['Age'].mean()

    elif not kk.empty:

        test.at[index,'Age'] = kk['Age'].mean()

#         train.ix[index] = kk['Age'].mean()

    elif not pp.empty:

#         train.ix[index] = pp['Age'].mean()

        test.at[index,'Age'] = pp['Age'].mean()

    else : 

        test.at[index,'Age'] = test['Age'].mean()

    

# for item in train[train['Age']==-1]:

#     print(item['Survived'])

# for tr in train:

#     print(tr)

# null 값 없는지 확인

print(train[train['Age']==-1])

print(test[test['Age']==-1])

# train = train.drop(train[train['Age']==-1].index)

print(train[train['Age']==-1]['Age'])

print(train[train['Age']==-1].index)
len(train.index)
train[train['Age']==-1].empty
train.describe(include = 'all')
train['FareBand'] = pd.qcut(train['Fare'],4,labels = [1,2,3,4])

test['FareBand'] = pd.qcut(test['Fare'],4,labels = [1,2,3,4])



# train = train.drop(['Fare'],axis = 1)

# test = test.drop(['Fare'],axis = 1)
train.head()
target = train['Survived']

train_data = train.drop('Survived',axis = 1)
train.info()
from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

import numpy as np
from sklearn.model_selection import KFold 

from sklearn.model_selection import cross_val_score
k_fold = KFold(n_splits = 10, shuffle = True, random_state = 0)

k_fold
clf = RandomForestClassifier(n_estimators=13)

clf
score = cross_val_score(clf,train_data,target,cv = k_fold, n_jobs = 1, scoring = 'accuracy')
score.mean()