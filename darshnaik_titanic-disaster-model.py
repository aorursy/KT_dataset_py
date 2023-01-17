# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import random as rnd



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier





import seaborn as sns

import matplotlib.pyplot as plt



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

# gender_sub = pd.read_csv('../input/gender_submission.csv')
train.head()
train.describe()
train.describe(include=['O'])
train.info()
len(train) - train.count()

# or train.isnull().sum
test.describe(include='O')
test.info()
len(test) - test.count()
survived = train[train.Survived == 1]

# or survived = train[train['Survived'] == 1]

# print(survived)

died = train[train['Survived'] == 0]



survived_perc = float(len(survived)/len(train)*100)

died_perc = float(len(died)/len(train)*100)



print("People Survived = %.2f%%" % survived_perc)

print("People died = %.2f%%" % died_perc)

train.Pclass.value_counts()
train.groupby('Pclass').Survived.value_counts()
train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean()
train.groupby('Pclass').Survived.mean().plot(kind='bar')
# Survival by Gender

train.groupby('Sex').Survived.value_counts()
train[['Sex','Survived']].groupby(['Sex']).mean()
train.groupby('Sex').Survived.mean().plot(kind='bar')
train.Embarked.value_counts()
train[['Embarked','Survived']].groupby(['Embarked']).mean()
train.groupby('Embarked').Survived.mean().plot(kind='bar')
train.Parch.value_counts()
train.groupby(['Parch']).Survived.mean().plot(kind='bar')
train[['SibSp','Survived']].groupby(['SibSp']).mean()
# similarly visualising survival with sibsp

train.groupby('SibSp').Survived.mean().plot(kind='bar')
# Visualizing age for survival

grph = plt.figure(figsize=(15,5))

gr1 = grph.add_subplot(131)

gr2 = grph.add_subplot(132)

gr3 = grph.add_subplot(133)



sns.violinplot(x="Embarked", y="Age", hue="Survived", data=train, split=True, ax=gr1)

sns.violinplot(x="Pclass", y="Age", hue="Survived", data=train, split=True, ax=gr2)

sns.violinplot(x="Sex", y="Age", hue="Survived", data=train, split=True, ax=gr3)
male_survived = train[(train['Survived']==1) & (train['Sex']=="male")]

female_survived = train[(train['Survived']==1) & (train['Sex']=="female")]

male_died = train[(train['Survived']==0) & (train['Sex']=="male")]

female_died = train[(train['Survived']==0) & (train['Sex']=="female")]



plt.figure(figsize=[15,5])

plt.subplot(111)

sns.distplot(survived['Age'].dropna().values, bins=range(0, 81, 1), kde=False, color='blue')

sns.distplot(died['Age'].dropna().values, bins=range(0, 81, 1), kde=False, color='red', axlabel='Age')



plt.figure(figsize=[15,5])



plt.subplot(121)

sns.distplot(female_survived['Age'].dropna().values, bins=range(0, 81, 1), kde=False, color='blue')

sns.distplot(female_died['Age'].dropna().values, bins=range(0, 81, 1), kde=False, color='red', axlabel='Female Age')



plt.subplot(122)

sns.distplot(male_survived['Age'].dropna().values, bins=range(0, 81, 1), kde=False, color='blue')

sns.distplot(male_died['Age'].dropna().values, bins=range(0, 81, 1), kde=False, color='red', axlabel='Male Age')
# Combining train and test set

both = [train,test]



for dataset in both:

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.')
train.head()
pd.crosstab(train['Title'], train['Sex'])
for dataset in both:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col', \

 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Other')



    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    

train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Other": 5}

for dataset in both:

    dataset['Title'] = dataset['Title'].map(title_mapping)

    dataset['Title'] = dataset['Title'].fillna(0)
train.head()
for dataset in both:

    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
train.head()
train.Embarked.unique()
train.Embarked.value_counts()
for dataset in both:

    dataset['Embarked'] = dataset['Embarked'].fillna('S')
for dataset in both:

    #print(dataset.Embarked.unique())

    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
train.head()
for dataset in both:

    age_avg = dataset['Age'].mean()

    age_std = dataset['Age'].std()

    age_null_count = dataset['Age'].isnull().sum()

    

    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)

    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list

    dataset['Age'] = dataset['Age'].astype(int)

    

train['AgeBand'] = pd.cut(train['Age'], 5)



print (train[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean())
train.head()
for dataset in both:

    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4
train.head()
for dataset in both:

    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())
for dataset in both:

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3

    dataset['Fare'] = dataset['Fare'].astype(int)
train.head()
for dataset in both:

    dataset['FamilySize'] = dataset['SibSp'] +  dataset['Parch'] + 1



print (train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean())
for dataset in both:

    dataset['IsAlone'] = 0

    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

    

print (train[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean())
train.head()
test.head()
features_drop = ['Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'FamilySize']

train = train.drop(features_drop, axis=1)

test = test.drop(features_drop, axis=1)
train.head()
train_try = train.copy()

test_try = test.copy()
train_try.head()
test_try.head()
train_try = train.drop("Title", axis=1)

test_try = test.drop("Title", axis=1)
test_try.head()
X_train = train_try.drop('Survived', axis=1)

y_train = train['Survived']

X_test = test_try.copy()



X_train.shape, y_train.shape, X_test.shape
X_train = X_train.drop("AgeBand", axis=1)
clf = SVC()

clf.fit(X_train, y_train)

y_pred_svc = clf.predict(X_test)

acc_svc = round(clf.score(X_train, y_train) * 100, 2)

print (acc_svc)
logreg = LogisticRegression()

logreg.fit(X_train, y_train)

Y_pred = logreg.predict(X_test)

acc_log = round(logreg.score(X_train, y_train) * 100, 2)

acc_log
# submission = pd.DataFrame({

#         "PassengerId": test["PassengerId"],

#         "Survived": y_pred_svc

#     })

# submission.to_csv('submission.csv', index=False)
submission_df = pd.DataFrame(columns=['PassengerId', 'Survived'])

submission_df['PassengerId'] = test['PassengerId']

submission_df['Survived'] = Y_pred

submission_df.to_csv('submissions.csv', header=True, index=False)

submission_df.head(100)