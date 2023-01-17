import pandas as pd

import matplotlib as plt

import re
titanic_train=pd.read_csv('../input/train.csv',header=0)

titanic_test=pd.read_csv('../input/test.csv',header=0)
dataset=titanic_train.append(titanic_test,sort=False)
dataset.head()
dataset.info()
dataset['Fare_1']=dataset['Fare'].fillna(value= 8.05)
dataset.Embarked.mode()
dataset['Embarked_1']=dataset['Embarked'].fillna(value='S')
dataset.isnull().sum()
dataset['Title'] = dataset['Name'].str.split(',',expand=True)[1].str.split('.',expand=True)[0].str.strip()
dataset.groupby('Title').count()
dataset['PassengerId'].groupby(dataset['Title']).count()
age_mean = dataset.groupby('Title')['Age'].mean()

age_median = dataset.groupby('Title')['Age'].median()
title_list=list(dataset['Title'].drop_duplicates())
dataset['Age_1'] = dataset['Age']
dataset.Age_1.isnull()
for i in title_list:

    dataset.loc[dataset.Age_1.isnull()==True,'Age_1']=dataset.loc[dataset.Title==i,'Age_1'].median()
dataset[dataset['Age'].isnull()].groupby(dataset[dataset['Age'].isnull()].Title)['Title'].count()
dataset.groupby(dataset.Title)['Age'].median()
dataset.loc[dataset.Age.isnull()==True,'Age_1'].mean()
dataset['Ticket'].nunique()
dataset.groupby('Ticket')['Ticket'].count()>1
dataset.loc[dataset['Ticket'].isin(dataset.groupby('Ticket')['Ticket'].count()>1),'Ticket']
dataset[dataset['Age']<1]
df = pd.crosstab(dataset.Age_1,dataset.Survived)

df['percent'] = df[1]/(df[0]+df[1])*15

df['total']=(df[1]+df[0])*1.0

import matplotlib.pyplot as plt

plt.plot(df.index,df.total)

plt.plot(df.index,df.percent)

plt.show()
plt.hist(dataset.Age_1,bins=20)

plt.show()
age_orig=dataset.Age.dropna()

plt.hist(age_orig,bins=20)

plt.show()
age_mr=dataset[dataset.Title=='Mr'].Age.dropna()

plt.hist(age_mr,bins=20)

plt.show()
dataset['AgeGroup']=(dataset['Age_1']/5).astype(int)
def YorN(x):

    if x > 0:

        return 1

    else:

        return 0

dataset['Sib_1'] = dataset['SibSp'].apply(lambda x: YorN(x))
dataset['Par_1']=dataset['Parch'].apply(lambda x: YorN(x))
dataset['Gender']=dataset['Sex'].apply(lambda x:1 if x=='male' else 0)
dataset['Gender'].head()
dataset['Cabin_1']=dataset['Cabin']

dataset['Cabin_1']=dataset['Cabin_1'].fillna(value=0)
dataset['Cabin_1']=dataset['Cabin_1'].apply(lambda x: 1 if x!=0 else x)
dataset.groupby(dataset.Cabin_1)['Cabin_1'].count()
pd.crosstab(dataset['Cabin_1'],dataset.Cabin)
dataset['Pclass_1']=dataset['Pclass']
plt.hist(dataset.Fare.dropna(),bins=20)

plt.show()
dataset['Fare_1']=dataset['Fare'].fillna(value=dataset['Fare'].median())
data_re=dataset.loc[:,'PassengerId':'Pclass_1']
data_re=data_re.drop('SibSp',axis=1)
def get_plot(feature,dataset):

    Sur_1=dataset[feature][dataset.Survived== 0].value_counts()

    Sur_2=dataset[feature][dataset.Survived== 1].value_counts()

    df=pd.DataFrame({u'Saved':Sur_1, u'Failed':Sur_2})

    df.plot(kind='bar', stacked=True)

    plt.title(u"Suvival by "+ feature)

    plt.xlabel(feature)

    plt.ylabel(u"Number")

    plt.show()
get_plot('SibSp',titanic_train)
data_new = dataset.copy()
data_new = data_new.drop('Name',axis=1)
data_new.Parch[data_new.Parch== 0].value_counts()
plt.scatter(data_new['Parch'],data_new['SibSp'])

plt.show()
pd.crosstab(data_new['Parch'],data_new['SibSp'])
from sklearn.linear_model import LogisticRegression
test=pd.get_dummies(data_re.AgeGroup,drop_first=True)
predictors = ['Fare_1','Embarked_1','AgeGroup','Sib_1','Par_1','Gender','Cabin_1','Pclass_1']

dummies = pd.get_dummies(data_re[predictors],drop_first=True)

dummies=pd.concat([dummies,test],axis=1)
x = dummies[:891]
x.describe()
y=data_re.Survived[:891]
regression = LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
regression.fit(x,y)
test=dummies[891:]
pre = regression.predict(test)
ytest=regression.predict(x)
pd.crosstab(ytest,y)
import numpy as np

preId =data_re.PassengerId[891:]

outcom = pd.DataFrame({'PassengerId':preId,'Survived':pre.astype(np.int32)})
outcom.to_csv('submission_regression.csv',index=False)