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
import seaborn as sns

import matplotlib.pyplot as plt

import scipy as sp
train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')
plt.style.use('fivethirtyeight')
train.head()
train.tail()
train.describe()
train[train['Sex'] == 'male'].shape
train[train['Sex'] == 'female'].shape
print(train[train['Survived'] == 1].shape[0]/len(train))

print(train[train['Survived'] == 0].shape[0]/len(train))
train.isna().sum()/len(train)
test.isna().sum()
Data_set = pd.concat([train, test])

Data_set.shape
train.head(5)
sns.countplot('Survived', data=train)
sns.countplot('Embarked', data=train)

plt.title('Embarked Train')
sns.countplot('Embarked', data=test)

plt.title('Embarked Test')
sns.countplot('Embarked',data=Data_set)
Data_set.groupby(['Sex','Survived'])['Survived'].count()
f, ax = plt.subplots(1,2, figsize=(18,8))

Data_set[['Sex','Survived']].groupby(['Sex']).mean().plot.bar(ax=ax[0])

ax[0].set_title('Survived vs Sex')

sns.countplot('Sex', hue='Survived', data=Data_set, ax=ax[1])

ax[1].set_title('Sex:Survived vs Dead')

plt.show()
pd.crosstab(Data_set.Pclass, Data_set.Survived, margins=True).style.background_gradient(cmap='summer_r')
fig, ax = plt.subplots(1,2, figsize=(18,8))

Data_set['Pclass'].value_counts().plot.bar(ax=ax[0])

ax[0].set_title('Number Of Passengers By Pclass')

ax[0].set_ylabel('Count')

sns.countplot('Pclass', hue='Survived', data=Data_set, ax=ax[1])

ax[1].set_title('Pclass:Survived vs Dead')
Data_set.groupby(['Pclass','Survived'])['Survived'].count()[3][0]/(Data_set.groupby(['Pclass','Survived'])['Survived'].count()[3][0]+\

                                                                   Data_set.groupby(['Pclass','Survived'])['Survived'].count()[3][1])
pd.crosstab([Data_set.Sex, Data_set.Survived], Data_set.Pclass, margins=True).style.background_gradient('summer_r')
sns.factorplot('Pclass','Survived', hue='Sex', data=Data_set)
Data_set.Age.max()
Data_set.Age.min()
Data_set.Age.mean()
f, ax = plt.subplots(1,2, figsize = (18,8))

sns.violinplot('Pclass','Age', hue = 'Survived', data = Data_set, split = True, ax = ax[0])

ax[0].set_title('Pclass and Age vs Survived')

ax[0].set_yticks(range(0,110,10))

sns.violinplot('Sex','Age', hue = 'Survived', data=Data_set, split=True, ax=ax[1])

ax[1].set_title('Sex and Age vs Survived')

ax[1].set_yticks(range(0,110,10))

Data_set['Initial'] = 0

for i in Data_set:

    Data_set['Initial'] = Data_set.Name.str.extract('([A=-Za-z]+)\.')
train['Initial'] = 0

for i in train:

    train['Initial'] = train.Name.str.extract('([A=-Za-z]+)\.')
pd.crosstab(Data_set.Initial, Data_set.Sex).T.style.background_gradient(cmap = 'summer_r')
Data_set['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don','Dona'],\

                        ['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr','Mrs'],inplace=True)
train['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don'],\

                        ['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr'],inplace=True)
Data_set.groupby('Initial')['Age'].mean()

Data_set.loc[(Data_set.Age.isnull())&(Data_set.Initial == 'Mr'), 'Age'] = 33

Data_set.loc[(Data_set.Age.isnull())&(Data_set.Initial == 'Master'), 'Age'] = 5

Data_set.loc[(Data_set.Age.isnull())&(Data_set.Initial == 'Miss'), 'Age'] = 22

Data_set.loc[(Data_set.Age.isnull())&(Data_set.Initial == 'Mrs'), 'Age'] = 37

Data_set.loc[(Data_set.Age.isnull())&(Data_set.Initial == 'Ohter'), 'Age'] = 45

Data_set.loc[(Data_set.Age.isnull())&(Data_set.Initial == 'Dona'), 'Age'] = 39
Data_set.Age.isnull().any()
f, ax = plt.subplots(1,2, figsize = (20,10))

Data_set[Data_set['Survived']==0].Age.plot.hist(ax=ax[0], bins=20, edgecolor='black',color='red')

ax[0].set_title('Survived = 0')

x1 = list(range(0,85,5))

ax[0].set_xticks(x1)

Data_set[Data_set['Survived']==1].Age.plot.hist(ax=ax[1], bins=20, edgecolor='black',color='green')

ax[1].set_title('Survived = 1')

x2 = list(range(0,85,5))

ax[1].set_xticks(x2)
sns.factorplot('Pclass', 'Survived', col='Initial', data=Data_set)
pd.crosstab([Data_set.Embarked, Data_set.Pclass], [Data_set.Sex, Data_set.Survived], \

            margins=True).style.background_gradient(cmap='summer_r')
sns.factorplot('Embarked','Survived', data = Data_set)

fig = plt.gcf()

fig.set_size_inches(5,3)

f, ax = plt.subplots(2,2, figsize = (20,15))

sns.countplot('Embarked',data=Data_set, ax= ax[0,0])

ax[0,0].set_title('# of Passengers Boarded')

sns.countplot('Embarked', hue = 'Sex',data=Data_set, ax=ax[0,1])

ax[0,1].set_title('Male Female split for Embarked')

sns.countplot('Embarked', hue = 'Survived', data=Data_set, ax=ax[1,0])

ax[1,0].set_title('Embarked vs Survived')

sns.countplot('Embarked', hue='Pclass', data=Data_set, ax=ax[1,1])

ax[1,1].set_title('Embarked vs Pclass')

plt.subplots_adjust(wspace=0.2, hspace=0.5)
sns.factorplot('Pclass','Survived', hue = 'Sex', col = 'Embarked', data=Data_set)
Data_set.Embarked.isna().sum()
Data_set.Embarked.fillna('S', inplace=True)
Data_set.Embarked.isnull().any()
pd.crosstab([Data_set.SibSp], Data_set.Survived).style.background_gradient(cmap='summer_r')
f, ax = plt.subplots(1,2, figsize=(20,8))

sns.barplot('SibSp','Survived', data=Data_set, ax= ax[0])

ax[0].set_title('SibSp vs Survived')

sns.factorplot('SibSp', 'Survived', data=Data_set, ax=ax[1])

ax[1].set_title('SibSp vs Survived')

plt.close(2)
pd.crosstab(Data_set.SibSp, Data_set.Pclass).style.background_gradient(cmap='summer_r')
pd.crosstab(Data_set.Parch, Data_set.Pclass).style.background_gradient(cmap='summer_r')
f, ax = plt.subplots(1,2,figsize=(20,8))

sns.barplot('Parch', 'Survived', data=Data_set, ax=ax[0])

ax[0].set_title('Parch vs Survived')

sns.factorplot('Parch', 'Survived', data=Data_set, ax=ax[1])

ax[1].set_title('Parch vs Survived')

plt.close(2)

print('Highest Fare was:',Data_set['Fare'].max())

print('Lowest Fare was:',Data_set['Fare'].min())

print('Average Fare was:',Data_set['Fare'].mean())
Data_set.Fare.isna().sum()
test[test.Fare.isna() == True]
train[(train.Embarked == 'S')&(train.Age > 58)&(train.Age < 64)].Fare.mean()
Data_set.Fare.fillna(26, inplace=True)
Data_set.isna().sum()
f, ax = plt.subplots(1,3, figsize=(20,8))

sns.distplot(Data_set[Data_set['Pclass']==1].Fare, ax=ax[0])

ax[0].set_title('Fares of Pclass 1')

sns.distplot(Data_set[Data_set['Pclass']==2].Fare, ax=ax[1])

ax[1].set_title('Fares of Pclass 2')

sns.distplot(Data_set[Data_set['Pclass']==3].Fare, ax=ax[2])

ax[2].set_title('Fares of Pclass 3')
sns.heatmap(Data_set.corr(),annot=True,cmap='RdYlGn',linewidths=0.2) #data.corr()-->correlation matrix

fig=plt.gcf()

fig.set_size_inches(10,8)

plt.show()
Data_set.head()
Data_set.Age
facet = sns.FacetGrid(Data_set, hue="Survived",aspect=4)

facet.map(sns.kdeplot,'Age',shade= True)

facet.add_legend()
pd.qcut(Data_set.Age, 5)
Data_set['Age_band'] = 0

Data_set.loc[Data_set['Age']<= 21, 'Age_band'] = 0

Data_set.loc[(Data_set['Age']>21)&(Data_set['Age']<=26), 'Age_band'] = 1

Data_set.loc[(Data_set['Age']>26)&(Data_set['Age']<=33), 'Age_band'] = 2

Data_set.loc[(Data_set['Age']>33)&(Data_set['Age']<=39), 'Age_band'] = 3

Data_set.loc[(Data_set['Age']>39)&(Data_set['Age']<=50), 'Age_band'] = 4

Data_set.loc[Data_set['Age'] > 50, 'Age_band'] = 5
Data_set['Age_band'].value_counts().to_frame().style.background_gradient(cmap='summer')
sns.factorplot('Age_band', 'Survived', data=Data_set, col='Pclass')
Data_set['Family_size'] = 0

Data_set['Family_size'] = Data_set['Parch'] + Data_set['SibSp']

Data_set['Alone'] = 0

Data_set.loc[Data_set.Family_size ==0, 'Alone'] = 1
f, ax = plt.subplots(1,2, figsize=(18,6))

sns.factorplot('Family_size', 'Survived', data=Data_set, ax=ax[0])

ax[0].set_title('Family_Size vs Survived')

sns.factorplot('Alone', 'Survived', data=Data_set, ax=ax[1])

ax[1].set_title('Alone vs Survived')

plt.close(2)

plt.close(3)
sns.factorplot('Alone', 'Survived', data=Data_set, hue='Sex', col='Pclass')
Data_set['Fare_range'] = pd.qcut(Data_set['Fare'], 6)

Data_set.groupby(['Fare_range'])['Survived'].mean().to_frame().style.background_gradient('summer_r')
Data_set['Fare_cat'] = 0

Data_set.loc[Data_set['Fare']<= 7.775, 'Fare_cat'] = 0

Data_set.loc[(Data_set['Fare']>7.775)&(Data_set['Fare']<=8.662) ,'Fare_cat'] = 1

Data_set.loc[(Data_set['Fare']>8.662)&(Data_set['Fare']<=14.454),'Fare_cat'] = 2

Data_set.loc[(Data_set['Fare']>14.454)&(Data_set['Fare']<=26),'Fare_cat'] = 3

Data_set.loc[(Data_set['Fare']>26)&(Data_set['Fare']<=53.1),'Fare_cat'] = 4

Data_set.loc[Data_set['Fare']>53.1,'Fare_cat'] = 5
sns.factorplot('Fare_cat', 'Survived', data=Data_set, hue='Sex')
Data_set.loc[414]
Data_set.loc[414,'Initial']
Data_set['Initial'].unique()
Data_set['Sex'].replace(['male','female'], [0,1], inplace=True)

Data_set['Embarked'].replace(['S','C','Q'], [0,1,2], inplace=True)

Data_set['Initial'].replace(['Mr','Mrs','Miss','Master','Other'], [0,1,2,3,4],inplace=True)
Data_set.head(1)
Data_set.drop(['Name','Age','Ticket','Fare','Cabin','Fare_range'],axis=1,inplace=True)
sns.heatmap(Data_set.corr(),annot=True,cmap='RdYlGn',linewidths=0.2,annot_kws={'size':20})

fig=plt.gcf()

fig.set_size_inches(18,15)

plt.xticks(fontsize=14)

plt.yticks(fontsize=14)

plt.show()
from sklearn.linear_model import LogisticRegression #logistic regression

from sklearn.ensemble import RandomForestClassifier #Random Forest

from sklearn.model_selection import train_test_split #training and testing data split

from sklearn.model_selection import StratifiedShuffleSplit

from sklearn import metrics #accuracy measure

from sklearn.metrics import confusion_matrix #for confusion matrix
test_done = Data_set[(Data_set['Survived'] != 0)&(Data_set['Survived'] != 1)]
del test_done['Survived']
train_done = Data_set[(Data_set['Survived'] == 0)|(Data_set['Survived'] == 1)]
train_done
del train_done['PassengerId']
target = train_done['Survived']
split = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=42)

for trn_idx, tst_idx in split.split(train_done, target):

    sampled_trn = train_done.loc[trn_idx]

    sampled_tst = train_done.loc[tst_idx]
len(sampled_tst)
lr_clf = LogisticRegression()

rf_clf = RandomForestClassifier()
sampled_trn
smp_target = sampled_trn['Survived']

del sampled_trn['Survived']
lr_clf.fit(sampled_trn, smp_target)
tst_target = sampled_tst['Survived']

del sampled_tst['Survived'] 
pred = lr_clf.predict(sampled_tst)
print('Accuracy for Logistic Regression is',metrics.accuracy_score(pred,tst_target))
sampled_trn
Pid = test_done['PassengerId']

del test_done['PassengerId']
ans = pd.Series(lr_clf.predict(test_done))
ans = ans.astype('int')
submission = pd.concat([Pid, ans], axis=1)
submission.columns = [['PassengerId','Survived']]
submission.to_csv('submission_stratified.csv', index=False)
pd.read_csv('submission_stratified.csv')
train, test = train_test_split(train_done, test_size = 0.2, random_state=42,\

                                                    stratify=train_done['Survived'])

train_X=train[train.columns[1:]]

train_Y=train[train.columns[:1]]

test_X=test[test.columns[1:]]

test_Y=test[test.columns[:1]]

X=train_done[train_done.columns[1:]]

Y=train_done['Survived']
Data_set.head()
test_set = Data_set[(Data_set.Survived != 0)&(Data_set.Survived != 1)]
Data_set.shape
test_set.shape
train_set = Data_set[(Data_set.Survived == 0)|(Data_set.Survived == 1)]
train_set.shape
target = train_set['Survived']

pid = test_set['PassengerId']
target
train_set.head()
del train_set['PassengerId']

del train_set['Survived']
x_train, x_test, y_train, y_test = train_test_split(train_set, target, test_size = 0.3, random_state = 42, stratify=target)
lr_clf.fit(x_train, y_train)
pred = lr_clf.predict(x_test)
metrics.accuracy_score(pred, y_test)
x_train.head()
y_test
test_done.head().shape
ans = pd.Series(lr_clf.predict(test_done))
ans = ans.astype('int')
submission = pd.DataFrame( {

    'PassengerId' : test_set['PassengerId'],

    'Survived' : ans,

})
submission.to_csv('submission_trn_tst_split.csv', index=False)
pd.read_csv('submission_trn_tst_split.csv')
submiss.head()
submiss.to_csv('submission_test.csv')
pd.read_csv('submission_test.csv')