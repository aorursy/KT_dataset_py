import os

for dirname, _, filenames in os.walk('/kaggle/input/'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score, roc_auc_score

from sklearn.metrics import confusion_matrix

from sklearn.ensemble import RandomForestClassifier

from collections import Counter
#load train_set, test_set

train=pd.read_csv('/kaggle/input/titanic/train.csv')

test=pd.read_csv('/kaggle/input/titanic/test.csv')
#Make merged data with train, test data

df = pd.concat([train,test]).set_index('PassengerId')
# features and survived

f, ax = plt.subplots(1,3,figsize=(12,5))

sns.countplot('Pclass',hue='Survived',data=df,ax=ax[0])

ax[0].set_title('Pclass/Survived')

sns.countplot('Sex', hue='Survived', data=df, ax=ax[1])

ax[1].set_title('Sex/Survived')

sns.countplot('Embarked', hue='Survived', data=df, ax=ax[2])

ax[2].set_title('Embarked/Survived')
# age / survived (train set)

f, ax = plt.subplots(1,2,figsize=(12,5))

ax[0].hist(train[train['Survived']==0]['Age'], bins=30)

ax[0].set_ylim([0,70])

ax[0].set_title('dead/age')

ax[1].hist(train[train['Survived']==1]['Age'], bins=30, color='orange')

ax[1].set_ylim([0,70])

ax[1].set_title('alive/age')
# find null in data

df.isnull().sum()
# fill Fare with Pclass.mean

# because Fare is very relevant with Pclass.

df.Fare = df.Fare.fillna(df.Fare.loc[df.Pclass==3].median())
# fill Embarked with most frequent value

Counter(df.Embarked)
df.Embarked = df.Embarked.fillna('S')
# find highest correlation(with pearson)

df.Sex=[{'male':0, 'female':1}[i] for i in df.Sex]

df[['Age','Fare','Pclass','Sex','SibSp','Survived','Sex']].corr(method='pearson')
# fill median Age of Pclass

age = list()

for i in df.index:

    if df.Age.isnull()[i]:

        s = df['Pclass'][i]

        age.append(df['Age'].loc[df['Pclass']== s].median())

    else : age.append(df['Age'][i])

df.Age=age
# Cabinet start with

Cabin=df.Cabin.fillna('0')

Cabin_start=[i[0] for i in Cabin]

df.Cabin = Cabin_start
f, ax = plt.subplots(1,2,figsize=(12,5))

sns.countplot('Cabin', hue='Survived', data=df, ax=ax[0])

sns.countplot('Cabin', hue='Pclass', data=df, ax=ax[1])
# A~F are like deck name which sorted by floor of the ship  A:top ~ G:bottom

# Sort Cabin to {A : Promenade / B,C : Bridge, Shelter / D,E : Salon, Upper /  F,G: Middle, Lower}

def Cabin_cla(x):

    if x=='A' : return 4

    elif x=='B'or'C': return 3

    elif x=='D'or'D': return 2

    elif x=='F'or'G': return 1

    else : return 0

df['Cabin']=list(map(Cabin_cla,df.Cabin))
# Same Ticket means companions (family or friends)

df.sort_values(['Ticket']).head()
# Companions number

df['Companions']=[dict(Counter(df.Ticket))[i] for i in df.Ticket]

df=df.drop('Ticket',axis=1)

plt.figure()

sns.countplot('Companions', hue='Survived', data=df)
# Name classification

title = [i.split(' ')[1].split('.')[0].strip() for i in df.Name]

Counter(title)
df.Name=title

plt.figure(figsize=(20,5))

sns.countplot('Name', hue='Survived', data=df)
# Mr, Mrs, Miss, master are only meaningfull features

# decide return value with linear survived proportion



def Name_cla(x):

    if x=='Mr': return 1

    elif x=='Master': return 3

    elif x=='Miss': return 4

    elif x=='Mrs': return 5

    else : return 2

df.Name = list(map(Name_cla,title))
plt.figure(figsize=(20,5))

sns.countplot('Name', hue='Survived', data=df)
# family (Parch, SibSp)

df['Family']=df['Parch']+df['SibSp']

f, ax = plt.subplots(1,3, figsize=(14,5))

sns.countplot('Parch', hue='Survived', data=df, ax=ax[0])

ax[0].set_title('Parch/Survived')

sns.countplot('SibSp', hue='Survived', data=df, ax=ax[1])

ax[1].set_title('SibSp/Survived')

sns.countplot('SibSp', hue='Survived', data=df, ax=ax[2])

ax[2].set_title('family/Survived')
# Parch and SibSp are simmilar with family

# drop Parch SibSp

df=df.drop(['Parch','SibSp'],axis=1)
# Set age level in order to predict overfitting 

# Because age is the most missing value(Assumed value)

def age_cla(x):

    if x <= 9 : return 0

    elif x <= 15 : return 1

    elif x <= 25 : return 2

    elif x <= 35 : return 3

    elif x <= 45 : return 4

    elif x <= 60 : return 5

    else : return 6

df.Age = list(map(age_cla, df.Age))



# age level distribution compare with train set

f, ax = plt.subplots(2,2,figsize=(12,10))

ax[0,0].hist(df[df['Survived']==0]['Age'], bins=30)

ax[0,0].set_ylim([0,500])

ax[0,0].set_xlim([-0.5,8])

ax[0,0].set_title('dead/age')

ax[0,1].hist(df[df['Survived']==1]['Age'], bins=30, color='orange')

ax[0,1].set_ylim([0,500])

ax[0,1].set_xlim([-0.5,8])

ax[0,1].set_title('alive/age')

ax[1,0].hist(train[train['Survived']==0]['Age'], bins=30)

ax[1,0].set_ylim([0,70])

ax[1,0].set_title('dead/age_train')

ax[1,1].hist(train[train['Survived']==1]['Age'], bins=30, color='orange')

ax[1,1].set_ylim([0,70])

ax[1,1].set_title('alive/age_train')
# str data to numeric

df.Embarked = [{'S':1, 'Q':2, 'C':3}[i] for i in df.Embarked]
#Set y_train, X_train, X_test

y_train=df.Survived.dropna()

X_train=df[df.Survived.notnull()].drop('Survived',axis=1)

X_test=df[df.Survived.isnull()].drop('Survived',axis=1)
X_train.head()
# fit the classifier with train data

# find max roc_auc_score

# find max_depth maxized roc_auc_score and CV_score

from sklearn.model_selection import cross_val_score

cv_score = list()

auc_score = list()

for i in [5,6,7,8,9]:

    rfc=RandomForestClassifier(criterion='gini',random_state=4, max_depth=i)

    rfc.fit(X_train, y_train)

    tree_predicted = rfc.predict(X_train)

    auc_score.append(roc_auc_score(y_train,tree_predicted))

    scores = cross_val_score(rfc, X_train, y_train,cv=5)

    cv_score.append(scores.mean())

df_score=pd.DataFrame({'max_depth':[5,6,7,8,9],'cv_score':cv_score,'auc_score':auc_score})
df_score
# to avoid overfitting

# Selecting max_depth = 8

rfc=RandomForestClassifier(criterion='gini',random_state=4, max_depth=8)

rfc.fit(X_train, y_train)

tree_predicted = rfc.predict(X_train)

roc_auc_score(y_train,tree_predicted)
# if I choose max_depth=8, final score is 0.78947

# it scores higer when max_depth=7, 

# I think this because score is sensitive for small changes, 

# and model with max_depth=8 could be little overfitting.



rfc=RandomForestClassifier(criterion='gini',random_state=4, max_depth=7)

rfc.fit(X_train, y_train)

tree_predicted = rfc.predict(X_train)

y_test = rfc.predict(X_test)
sum(y_test)
# Write the result to csv

PassengerId = list(X_test.reset_index()['PassengerId'])

result = pd.DataFrame({'PassengerId':PassengerId,'Survived':y_test})

result['Survived']=[int(i) for i in result.Survived]

result.to_csv('result.csv', index=False)