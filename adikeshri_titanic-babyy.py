# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns

sns.set()



pd.set_option('display.max_row',1000)

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train=pd.read_csv('../input/titanic/train.csv',index_col='PassengerId')

test=pd.read_csv('../input/titanic/test.csv',index_col='PassengerId')

train.info();test.info()
train.corr()
train.sample(5)
train['FamilySize']=train.SibSp+train.Parch

train.drop(['SibSp','Parch'],axis=1,inplace=True)

test['FamilySize']=test.SibSp+test.Parch+1

test.drop(['SibSp','Parch'],axis=1,inplace=True)

train.info();test.info()
x=train.Name.str.split(',')

first=[]

for i in x:

    first.append(i[1])

pre=[]

for i in first:

    pre.append(i.split()[0])

plt.figure(figsize=(14,8))

sns.countplot(pre)

plt.xticks(rotation=90)

plt.show()

train['prefix']=pre

train.info()
x=test.Name.str.split(',')

first=[]

for i in x:

    first.append(i[1])

pre=[]

for i in first:

    pre.append(i.split()[0])

plt.figure(figsize=(14,8))

sns.countplot(pre)

plt.xticks(rotation=90)

plt.show()

test['prefix']=pre
plt.figure(figsize=(14,8))

sns.distplot(train[train.prefix=='Mr.'].Age.dropna(),color='blue',label='Mr.',hist=False)

sns.distplot(train[train.prefix=='Mrs.'].Age.dropna(),color='maroon',label='Mrs.',hist=False)

sns.distplot(train[train.prefix=='Miss.'].Age.dropna(),color='black',label='Miss.',hist=False)

sns.distplot(train[train.prefix=='Master.'].Age.dropna(),color='grey',label='Master.',hist=False)
print('Median age of people with Mr. prefix: ' + str(train[train.prefix=='Mr.'].Age.median()))

print('Median age of people with Mrs. prefix: ' + str(train[train.prefix=='Mrs.'].Age.median()))

print('Median age of people with Miss. prefix: ' + str(train[train.prefix=='Miss.'].Age.median()))

print('Median age of people with Master. prefix: ' + str(train[train.prefix=='Master.'].Age.median()))

print('Median age of people with Don. prefix: ' + str(train[train.prefix=='Don.'].Age.median()))

print('Median age of people with Rev. prefix: ' + str(train[train.prefix=='Rev.'].Age.median()))

print('Median age of people with Dr. prefix: ' + str(train[train.prefix=='Dr.'].Age.median()))
train.loc[train.prefix=='Mr.','Age']=train.loc[train.prefix=='Mr.','Age'].fillna(30.0)

train.loc[train.prefix=='Mrs.','Age']=train.loc[train.prefix=='Mrs.','Age'].fillna(35.0)

train.loc[train.prefix=='Miss.','Age']=train.loc[train.prefix=='Miss.','Age'].fillna(21.0)

train.loc[train.prefix=='Master.','Age']=train.loc[train.prefix=='Master.','Age'].fillna(3.5)

train.loc[train.prefix=='Dr.','Age']=train.loc[train.prefix=='Dr.','Age'].fillna(46.5)



test.loc[test.prefix=='Mr.','Age']=test.loc[test.prefix=='Mr.','Age'].fillna(30.0)

test.loc[test.prefix=='Mrs.','Age']=test.loc[test.prefix=='Mrs.','Age'].fillna(35.0)

test.loc[test.prefix=='Miss.','Age']=test.loc[test.prefix=='Miss.','Age'].fillna(21.0)

test.loc[test.prefix=='Master.','Age']=test.loc[test.prefix=='Master.','Age'].fillna(3.5)

test.loc[test.prefix=='Dr.','Age']=test.loc[test.prefix=='Dr.','Age'].fillna(46.5)

train.info();test.info()
test.Age.fillna(21,inplace=True)

test[test.Age.isna()]
train.Age.isna().sum()
train[train.Embarked.isna()]
train.Embarked.fillna('S',inplace=True)

train.Embarked.unique()
train.Cabin=train.Cabin.str[0]

train.Cabin.sample(5)
train.Cabin.unique()
x=train['Name'].str.split()

lastname=[]

for i in x:

    lastname.append(i[-1])

count={}

for i in lastname:

    if i in count:

        count[i]+=1

    else:

        count[i]=1

for key in count.keys():

    if count[key]>1:

        print(key,count[key])
train[train.Pclass==3].Fare.median()
test.Fare.fillna(8.05,inplace=True)
train.info();test.info()
sns.countplot(train.Survived)
data=[train,test]

for dataset in data:

    dataset['Age'] = dataset['Age'].astype(int)

    dataset.loc[ dataset['Age'] <= 11, 'Age'] = 0

    dataset.loc[(dataset['Age'] > 11) & (dataset['Age'] <= 18), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 18) & (dataset['Age'] <= 22), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 22) & (dataset['Age'] <= 27), 'Age'] = 3

    dataset.loc[(dataset['Age'] > 27) & (dataset['Age'] <= 33), 'Age'] = 4

    dataset.loc[(dataset['Age'] > 33) & (dataset['Age'] <= 40), 'Age'] = 5

    dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 66), 'Age'] = 6

    dataset.loc[ dataset['Age'] > 66, 'Age'] = 6



for dataset in data:

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

    dataset.loc[(dataset['Fare'] > 31) & (dataset['Fare'] <= 99), 'Fare']   = 3

    dataset.loc[(dataset['Fare'] > 99) & (dataset['Fare'] <= 250), 'Fare']   = 4

    dataset.loc[ dataset['Fare'] > 250, 'Fare'] = 5

    dataset['Fare'] = dataset['Fare'].astype(int)



titles = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}



for dataset in data:

    # extract titles

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

    # replace titles with a more common title or as Rare

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr',\

                                            'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    # convert titles into numbers

    dataset['Title'] = dataset['Title'].map(titles)

    # filling NaN with 0, to get safe

    dataset['Title'] = dataset['Title'].fillna(0)



for dataset in data:

    dataset['Age_Class']= dataset['Age']* dataset['Pclass']



for dataset in data:

    dataset['Fare_Per_Person'] = dataset['Fare']/(dataset['FamilySize']+1)

    dataset['Fare_Per_Person'] = dataset['Fare_Per_Person'].astype(int)
to_drop=['Name','Ticket','Cabin','prefix']

for col in to_drop:

    train.drop(col,axis=1,inplace=True)

    test.drop(col,axis=1,inplace=True)

train.info();test.info()
to_cat=['Pclass','Sex','Embarked']

for col in to_cat:

    dummies=pd.get_dummies(train[col])

    train=pd.concat([train,dummies],axis=1)

    train.drop(col,axis=1,inplace=True)

    dummies=pd.get_dummies(test[col])

    test=pd.concat([test,dummies],axis=1)

    test.drop(col,axis=1,inplace=True)

train.info();test.info()
train.Fare.head(5)
X=train.drop('Survived',axis=1)

Y=train.Survived



from sklearn.ensemble import RandomForestClassifier

import xgboost as xgb

model_xgb=xgb.XGBClassifier(learning_rate=0.001,n_estimators=1000)

model_xgb.fit(X,Y)

model=RandomForestClassifier(criterion = "gini", 

                                       min_samples_leaf = 1, 

                                       min_samples_split = 10,   

                                       n_estimators=100, 

                                       max_features='auto', 

                                       oob_score=True, 

                                       random_state=1, 

                                       n_jobs=-1)

model.fit(X,Y)

model.score(X,Y),model_xgb.score(X,Y)
pred=model.predict(test)

l=pd.read_csv('../input/titanic/gender_submission.csv')

ans=pd.DataFrame({'PassengerId':l.PassengerId,'Survived':pred})

ans.to_csv('rf.csv',index=False)



pred=model_xgb.predict(test)

ans=pd.DataFrame({'PassengerId':l.PassengerId,'Survived':pred})

ans.to_csv('xgb.csv',index=False)