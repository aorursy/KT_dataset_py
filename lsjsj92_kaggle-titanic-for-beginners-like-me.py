# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os


# Any results you write to the current directory are saved as output.
print(os.listdir("../input"))
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.head()
train.shape
test.shape
train.info()

train.describe()
train.isnull().sum()
test.isnull().sum()
survived = train[train['Survived'] == 1]
not_survived = train[train['Survived'] == 0]
print("surv : %.1f" %(len(survived) / len(train) * 100))
print("not surv : %.1f " %(len(not_survived) / len(train) * 100))
plt.bar(['survive', 'not survive'], [len(survived), len(not_survived)])
plt.title("survival")
plt.show()
survived_number_by_sex = train[train['Survived']==1]['Sex'].value_counts()
not_survived_number_by_sex = train[train['Survived']==0]['Sex'].value_counts()
survived_number_by_sex
not_survived_number_by_sex
tmp = pd.DataFrame([survived_number_by_sex, not_survived_number_by_sex])
tmp.index = ['sur', 'notsur']
tmp.head()
plt.bar(['female', 'male'], [tmp['female']['sur'], tmp['male']['sur']])
plt.xlabel('sex')
plt.ylabel('number of survived')
plt.grid()
plt.show()
sns.barplot(x='Sex', y='Survived', data=train)
number_of_pclass = train['Pclass'].value_counts()
number_of_pclass
sns.barplot(x='Pclass', y='Survived', data=train)
sns.factorplot('Pclass', 'Survived', hue='Sex', data=train)
plt.show()
print(train['Age'].max())
print(train['Age'].min())
print(train['Age'].mean())
f, ax = plt.subplots(figsize=(12,8))
sns.violinplot("Pclass", "Age", hue="Survived", data=train, split=True)
ax.set_yticks(range(0, int(train['Age'].max())+10, 10))
plt.show()
f, ax = plt.subplots(figsize=(12,8))
sns.violinplot("Sex", "Age", hue='Survived', split=True, data=train)
ax.set_yticks(range(0, int(train['Age'].max() + 10), 10))
plt.show()
import re
p = re.compile('([A-Za-z]+)\.')
for cnt, value in enumerate(train['Name']):
    print(p.search(value).group())
    if cnt >= 5: break
train['Initial'] = ''
init = []
p = re.compile('([A-Za-z]+)\.')
for cnt, value in enumerate(train['Name']):
    init.append(p.search(value).group())
train['Initial'] = init
train['Initial'].head(10)
train['Initial'].unique()
train['Initial'].value_counts()
pd.crosstab(train.Initial, train.Sex).T.style.background_gradient(cmap='winter')
pre = ['Mr.', 'Miss.', 'Mrs.', 'Master.', 'Mlle.','Mme.','Ms.','Dr.','Major.','Lady.','Countess.','Jonkheer.','Col.','Rev.','Capt.','Sir.','Don.']
aft = ['Mr', 'Miss', 'Mrs', 'Master', 'Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr']
train['Initial'].replace(pre,aft,inplace=True)
train['Initial'].value_counts()
train.groupby('Initial')['Age'].head(1)
train.groupby('Initial')['Age'].mean()
train.loc[(train.Age.isnull()) & (train.Initial == 'Master'), 'Age'] = 5
train.loc[(train.Age.isnull()) & (train.Initial == 'Miss'), 'Age'] = 22
train.loc[(train.Age.isnull())&(train.Initial=='Mr'), 'Age'] = 33
train.loc[(train.Age.isnull()) & (train.Initial=='Mrs'), 'Age'] = 36
train.loc[(train.Age.isnull()) & (train.Initial == 'Other'), 'Age'] = 46
train['Age'].isnull().sum()
f, ax = plt.subplots(1, 2, figsize=(18,10))
train[train['Survived'] == 0]['Age'].plot.hist(ax=ax[0], bins=20, edgecolor='black', color='red')
ax[0].set_title('not survived')
x = list(range(0, 90, 5))
ax[0].set_xticks(x)

train[train['Survived'] == 1]['Age'].plot.hist(ax=ax[1], bins=20, edgecolor='black', color='blue')
ax[1].set_title('survived')
ax[1].set_xticks(x)

plt.show()
sns.factorplot('Pclass', 'Survived', col='Initial', data=train)
plt.show()
pd.crosstab([train['Embarked'], train['Pclass']], [train['Sex'], train['Survived']]).style.background_gradient(cmap='winter')
sns.factorplot('Embarked', 'Survived', data=train)
plt.show()
f, ax = plt.subplots(2,2, figsize=(18,15))
sns.countplot('Embarked', data=train, ax=ax[0,0])
ax[0,0].set_title('number of boarded ')

sns.countplot('Embarked', data=train, hue='Sex', ax=ax[0,1])

sns.countplot('Embarked', data=train, hue='Survived', ax=ax[1,0])

sns.countplot('Embarked', data=train, hue='Pclass', ax=ax[1,1])
plt.show()
sns.factorplot('Pclass', 'Survived', hue='Sex', col='Embarked', data=train)
plt.show()
train['Embarked'].fillna('S', inplace=True)
train['Embarked'].isnull().sum()
train['SibSp'].value_counts()
train.groupby('SibSp')['Survived'].value_counts()
pd.crosstab([train['SibSp']], train['Survived']).style.background_gradient(cmap='winter')
sns.barplot(x='SibSp', y='Survived', data=train)
train['Parch'].value_counts()
pd.crosstab(train['Parch'], train['Survived']).style.background_gradient(cmap='winter')
sns.barplot(x='Parch', y='Survived', data=train)
print(train['Fare'].max())
print(train['Fare'].min())
print(train['Fare'].mean())
f, ax = plt.subplots(1,3 , figsize=(20,8))
sns.distplot(train[train['Pclass'] == 1]['Fare'], ax=ax[0])
ax[0].set_title('Fares in Pclass 1')

sns.distplot(train[train['Pclass'] == 2]['Fare'], ax=ax[1])
ax[1].set_title('Fares in Pclass 2')

sns.distplot(train[train['Pclass'] == 3]['Fare'], ax = ax[2])
ax[2].set_title('Fares in Pclass 3')

plt.show()
plt.figure(figsize=(15,10))
sns.heatmap(train.drop('PassengerId', axis=1).corr(), annot=True, linewidths=0.2, cmap='PuBu')
del train
del test
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train_test = [train, test]
train_test[0].head()
print(train_test[0].shape)
print(train_test[1].shape)
for data in train_test:
    data['Initial'] = ''
    init = []
    p = re.compile('([A-Za-z]+)\.')
    for value in data['Name']:
        init.append(p.search(value).group())
    data['Initial'] = init
train.head()
pre = ['Mr.', 'Miss.', 'Mrs.', 'Master.', 'Mlle.','Mme.','Ms.','Dr.','Major.','Lady.','Countess.','Jonkheer.','Col.','Rev.','Capt.','Sir.','Don.', 'Dona.']
aft = ['Mr', 'Miss', 'Mrs', 'Master', 'Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr','Other']
for data in train_test:
    data['Initial'].replace(pre,aft,inplace=True)
train.head()
test.head()
train['Initial'].value_counts()
train.groupby('Initial')['Age'].head(1)
train.groupby('Initial')['Age'].mean()
for data in train_test:
    data.loc[(data['Age'].isnull()) & (data['Initial'] == 'Master'), 'Age'] = 5
    data.loc[(data['Age'].isnull()) & (data['Initial'] == 'Miss'), 'Age'] = 22
    data.loc[(data['Age'].isnull()) & (data['Initial'] == 'Mr'), 'Age'] = 33
    data.loc[(data['Age'].isnull()) & (data['Initial'] == 'Mrs'), 'Age'] = 36
    data.loc[(data['Age'].isnull()) & (data['Initial'] == 'Other'), 'Age'] = 46
train['Age'].isnull().sum()
test['Age'].isnull().sum()
mapping = {
    'Mr' : 1,
    'Miss' : 2,
    'Mrs' : 3,
    'Master' : 4,
    'Other' : 5
}

for data in train_test:
    data['Initial'] = data['Initial'].map(mapping).astype(int)
train.head()
train['Initial'].value_counts()
mapping ={
    'female' : 1,
    'male': 0
}
for data in train_test:
    data['Sex'] = data['Sex'].map(mapping).astype(int)
train['Sex'].value_counts()
train.head()
for data in train_test:
    data['Embarked'].fillna('S', inplace=True)
print(train['Embarked'].unique())
print(test['Embarked'].unique())
mapping = {
    'S' : 0,
    'C' : 1,
    'Q' : 2
}
for data in train_test:
    data['Embarked'] = data['Embarked'].map(mapping).astype(int)
for data in train_test:
    data.loc[data['Age'] <= 16, 'Age'] = 0
    data.loc[(data['Age'] > 16 ) & ( data['Age'] <= 32 ), 'Age'] = 1
    data.loc[(data['Age'] > 32 ) & ( data['Age'] <= 48 ), 'Age'] = 2
    data.loc[(data['Age'] > 48 ) & ( data['Age'] <= 64 ), 'Age'] = 3
    data.loc[data['Age'] > 64, 'Age'] = 4
    
train.head()
print(train['Age'].unique())
print(test['Age'].unique())
train['Age'] = train['Age'].astype(int)
test['Age'] = test['Age'].astype(int)
print(train['Age'].unique())
print(test['Age'].unique())
train['Age'].value_counts().to_frame().style.background_gradient('summer')
print(train['Fare'].isnull().sum())
print(test['Fare'].isnull().sum())
for data in train_test:
    data['Fare'].fillna(train['Fare'].median(), inplace=True)
print(test['Fare'].isnull().sum())
train['FareBand'] = pd.qcut(train['Fare'], 4) # four section.
train.groupby(['FareBand'])['Survived'].mean().to_frame().style.background_gradient('summer')
for data in train_test:
    data.loc[data['Fare'] <= 7.91, 'Fare'] = 0
    data.loc[(data['Fare'] > 7.91) & ( data['Fare'] <= 14.454 ), 'Fare'] = 1
    data.loc[(data['Fare'] > 14.454 ) & (data['Fare'] <= 31), 'Fare' ] = 2
    data.loc[data['Fare'] > 31, 'Fare'] = 3
train['Fare'] = train['Fare'].astype(int)
test['Fare'] = test['Fare'].astype(int)
print(train['Fare'].unique())
print(test['Fare'].unique())
for data in train_test:
    data['Family'] = data['SibSp'] + data['Parch']
train[['Family', 'Survived']].groupby(['Family']).mean()
for data in train_test:
    data.loc[data['Family'] == 0, 'Family'] = 0
    data.loc[(data['Family'] >= 1) & (data['Family'] < 4), 'Family'] = 1
    data.loc[(data['Family'] >= 4) & (data['Family'] < 7), 'Family'] = 2
    data.loc[(data['Family'] >= 7), 'Family'] = 3
train['Family'].unique()
train[['Family', 'Survived']].groupby(['Family']).mean()
test.head(2)
drop_list = ['Name', 'SibSp', 'Parch', 'Ticket', 'Cabin']
train = train.drop(drop_list, axis=1)
test = test.drop(drop_list, axis=1)

train = train.drop(['PassengerId', 'FareBand'], axis=1)
train.head()
test.head()



