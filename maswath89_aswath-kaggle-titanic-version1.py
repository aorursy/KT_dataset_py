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
import numpy as np

import pandas as pd

import seaborn as sns
df_train = pd.read_csv('train.csv',index_col = 'PassengerId')

df_train.head()
df_test2 = pd.read_csv('/kaggle/input/test.csv')

df_test2.head()
df_test = pd.read_csv('test.csv', index_col='PassengerId')

df_test.head()
df_test['Survived'] = 9999

dataset = pd.concat([df_train, df_test], axis = 0)

dataset.head()
dataset.info()
dataset.Embarked.value_counts()
pd.isnull(dataset).sum()
sns.barplot(x='Sex', y = 'Survived', data = df_train)
print("Females: ", df_train['Survived'][df_train['Sex'] == 'female'].value_counts(normalize = True)[1]*100)

print("Males: ", df_train['Survived'][df_train['Sex'] == 'male'].value_counts(normalize = True)[1]*100)
df_train['Age'] = df_train['Age'].fillna(-19)

df_test['Age'] = df_test['Age'].fillna(-19)
bins = [-20,0,5,12,18,24,35,60, np.inf]

labels = ['Unk','Infants','Children','Teens','Young','Adults','Old','Senior']

df_train['AgeGroup'] = pd.cut(df_train['Age'], bins, labels = labels)

df_test['AgeGroup'] = pd.cut(df_test['Age'], bins, labels = labels)

sns.barplot(x='AgeGroup',y = 'Survived', data = df_train)
df_train.AgeGroup.isnull().value_counts()
print("Pclass 1: ", df_train['Survived'][df_train.Pclass == 1].value_counts(normalize= True)[1]*100)

print("Pclass 2: ", df_train['Survived'][df_train.Pclass == 2].value_counts(normalize= True)[1]*100)

print("Pclass 3: ", df_train['Survived'][df_train.Pclass == 3].value_counts(normalize= True)[1]*100)

sns.barplot(x = 'Pclass', y = 'Survived', data =df_train)
df_train['SibSp'].value_counts()
print(df_train['Survived'][df_train['SibSp'] == 0].value_counts(normalize = True)[1]*100)

print(df_train['Survived'][df_train['SibSp'] == 1].value_counts(normalize = True)[1]*100)

print(df_train['Survived'][df_train['SibSp'] == 2].value_counts(normalize = True)[1]*100)

print(df_train['Survived'][df_train['SibSp'] == 3].value_counts(normalize = True)[1]*100)

print(df_train['Survived'][df_train['SibSp'] == 4].value_counts(normalize = True)[1]*100)

#print(df_train['Survived'][df_train['SibSp'] == 5].value_counts(normalize = True)[1]*100)

#print(df_train['Survived'][df_train['SibSp'] == 8].value_counts(normalize = True)[1]*100)

sns.barplot(x = 'SibSp' , y = 'Survived', data = df_train)
df_train['Parch'].value_counts()
sns.barplot(x = 'Parch' , y = 'Survived', data = df_train)
df_test.describe()
df_train = df_train.drop(['Cabin','Ticket'], axis = 1)

df_test = df_test.drop(['Cabin','Ticket'], axis = 1)
df_train.head()
df_train['Embarked'].value_counts()
df_train = df_train.fillna({'Embarked': 'S'})

df_test = df_test.fillna({'Embarked': 'S'})
#create a combined group of both datasets

combine = [df_train, df_test]



#extract a title for each Name in the train and test datasets

for dataset in combine:

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)



pd.crosstab(df_train['Title'], df_train['Sex'])
#replace various titles with more common names

for dataset in combine:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    

df_train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
#map each of the title groups to a numerical value

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Royal": 5, "Rare": 6}

for dataset in combine:

    dataset['Title'] = dataset['Title'].map(title_mapping)

    dataset['Title'] = dataset['Title'].fillna(0)



df_train.head()
mr_age = df_train[df_train['Title'] == 1]['AgeGroup'].mode()

miss_age = df_train[df_train["Title"] == 2]["AgeGroup"].mode()

mrs_age = df_train[df_train["Title"] == 3]["AgeGroup"].mode()

master_age = df_train[df_train["Title"] == 4]["AgeGroup"].mode()

royal_age = df_train[df_train["Title"] == 5]["AgeGroup"].mode()

rare_age = df_train[df_train["Title"] == 6]["AgeGroup"].mode() 
df_train['AgeGroup'].isnull().value_counts()
df_train['AgeGroup'].value_counts()
#map each Age value to a numerical value

age_mapping = {'Unk': 0, 'Infants': 1, 'Children': 2, 'Teens': 3, 'Young': 4, 'Adults': 5, 'Old': 6, 'Senior':7}

df_train['AgeGroup'] = df_train['AgeGroup'].map(age_mapping)

df_test['AgeGroup'] = df_test['AgeGroup'].map(age_mapping)



df_train.head()



#dropping the Age feature for now, might change

df_train = df_train.drop(['Age'], axis = 1)

df_test = df_test.drop(['Age'], axis = 1)
df_train.head()
df_train = df_train.drop(['Name'], axis =1)

df_test = df_test.drop(['Name'], axis = 1)
df_train['Sex'] = df_train['Sex'].map({'male' : 1, 'female' : 2})

df_test['Sex'] = df_test['Sex'].map({'male' : 1, 'female' : 2})
df_train['Embarked'] = df_train['Embarked'].map({'S' : 1, 'C' : 2, 'Q' : 3})

df_test['Embarked'] = df_test['Embarked'].map({'S' : 1, 'C' : 2, 'Q' : 3})
df_test[df_test.Fare.isnull()]
df_test.at[1044,'Fare']=13
#fill in missing Fare value in test set based on mean fare for that Pclass 

#df_test['Fare'].isnull()= df_train[df_train["Pclass"] == 3]["Fare"].mean()

        

#map Fare values into groups of numerical values

df_train['FareBand'] = pd.qcut(df_train['Fare'], 4, labels = [1, 2, 3, 4])

df_test['FareBand'] = pd.qcut(df_test['Fare'], 4, labels = [1, 2, 3, 4])



#drop Fare values

df_train = df_train.drop(['Fare'], axis = 1)

df_test = df_test.drop(['Fare'], axis = 1)
df_test = df_test.drop('Survived', axis = 1)
df_test.head()
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()



lr.fit(df_train.drop('Survived', axis = 1), df_train['Survived'])


df_test.head()
df_test = df_test.set_index('PassengerId')

from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()



dt.fit(df_train.drop('Survived', axis = 1), df_train['Survived'])

dt.predict(df_test)



y = pd.DataFrame()



y['PassengerId'] = df_test2['PassengerId']

y = dt.predict(df_test)

y = pd.DataFrame(y)



#df_test = df_test.reset_index('PassengerId')

ya = pd.DataFrame()

ya['PassengerId'] = df_test2['PassengerId']

ya['Survived'] = y

ya[['PassengerId','Survived']].to_csv('gender_submission.csv')
#df_test = df_test.set_index('PassengerId')

from sklearn.svm import LinearSVC

svc = LinearSVC()



svc.fit(df_train.drop('Survived', axis = 1), df_train['Survived'])



y = pd.DataFrame()



y['PassengerId'] = df_test2['PassengerId']

y = svc.predict(df_test)

y = pd.DataFrame(y)



#df_test = df_test.reset_index('PassengerId')

ya = pd.DataFrame()

ya['PassengerId'] = df_test2['PassengerId']

ya['Survived'] = y

ya[['PassengerId','Survived']].to_csv('gender_submission.csv')
# Gradient Boosting Classifier

from sklearn.ensemble import GradientBoostingClassifier



gbk = GradientBoostingClassifier()

gbk.fit(df_train.drop('Survived', axis = 1), df_train['Survived'])

y = pd.DataFrame()



y['PassengerId'] = df_test2['PassengerId']

y = svc.predict(df_test)

y = pd.DataFrame(y)



#df_test = df_test.reset_index('PassengerId')

ya = pd.DataFrame()

ya['PassengerId'] = df_test2['PassengerId']

ya['Survived'] = y

ya[['PassengerId','Survived']].to_csv('gender_submission.csv')
y = pd.DataFrame()

y['PassengerId'] = df_test['PassengerId']

y = lr.predict(df_test)

y = pd.DataFrame(y)
#df_test = df_test.reset_index('PassengerId')

ya = pd.DataFrame()

ya['PassengerId'] = df_test2['PassengerId']

ya['Survived'] = y

ya.to_csv('gender_submission.csv')
ya.head()