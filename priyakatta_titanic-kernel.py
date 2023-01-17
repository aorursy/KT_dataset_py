# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

from subprocess import check_output

print(os.listdir("../input"))

print(check_output(["ls", "../input"]).decode("utf8"))









# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import random as rnd



# imports for visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbours import KNeighborsclassifier

from sklearn.tree import DecisionTreeClassifier
#Reading input data

train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')
train_df.head()
train_df.tail()
train_df.info()

print('-'*40)

test_df.info()
train_df.isnull().sum()
classmeans=train_df.pivot_table('Fare',index=['Sex','Pclass'],aggfunc=np.mean)

print(classmeans)
import warnings

warnings.filterwarnings('ignore')

sns.set_style('darkgrid')

g=sns.FacetGrid(train_df,row='Pclass',height=3,aspect=1.8,palette='muted')

g.map(sns.pointplot,'Embarked','Survived','Sex')

g.add_legend()

plt.show()
grid=sns.FacetGrid(train_df,row="Sex",col="Survived",height=3,aspect=1.5)

grid.map(plt.hist,'Age',stacked=True,bins=20,linewidth=0.5)


plt.figure(figsize=(12,10))

plt.subplot(2,2,1)

sns.barplot('SibSp','Survived',data=train_df)

plt.subplot(2,2,2)

sns.barplot('Parch','Survived',data=train_df)

plt.subplot(2,2,3)

sns.barplot('Ticket','Survived',data=train_df)

plt.subplot(2,2,4)

sns.barplot('Cabin','Survived',data=train_df)



df=pd.concat([train_df,test_df],ignore_index=True,sort=False)

df.tail(10)
df.isnull().sum()[df.isnull().sum()>0]



df['Title']=df['Name'].str.split(',').str[1].str.split('.').str[0]

df['Title']=df['Title'].str.strip()
df['Title'].unique()

Title_list=df['Title'].unique().tolist()

print(Title_list)
df.head()
pd.crosstab(df['Title'],df['Sex'])
# Finding mean Fare with respect to title and sex

df.pivot_table('Age',index=['Title','Sex'],aggfunc=np.mean,margins=False)

map_title={

    'Capt':        'Officer',

    'Col':         'Officer',

    'Dr':          'Officer',

    'Major':       'Officer',

    'Rev':         'Officer',

    'Don':         'Royalty',

    'Dona':        'Royalty',

    'Jonkheer':    'Royalty',

    'Lady':        'Royalty',

    'Sir':         'Royalty',

    'the Countess':'Royalty',

    'Mlle':        'Miss',

    'Mme':         'Miss',

    'Ms':          'Mrs',

    'Mr':          'Mr',

    'Master':      'Master',

    'Miss':        'Miss',

    'Mrs':         'Mrs'}

df['Title']=df['Title'].map(map_title)
df.pivot_table('Age',index=['Title','Sex'],aggfunc=np.mean,margins=False)
def new_age (cols):

    title=cols[0]

    Sex=cols[1]

    Age=cols[2]

    if pd.isnull(Age):

        if title=='Master' and Sex=="male":

            return 5.48

        elif title=='Miss' and Sex=='female':

            return 21.77

        elif title=='Mr' and Sex=='male': 

            return 32.25

        elif title=='Mrs' and Sex=='female':

            return 36.99

        elif title=='Officer' and Sex=='female':

            return 49

        elif title=='Officer' and Sex=='male':

            return 46.14

        elif title=='Royalty' and Sex=='female':

            return 40.00

        else:

            return 42.33

    else:

        return Age 

df.Age=df[['Title','Sex','Age']].apply(new_age,axis='columns')
df['HasCabin']= ~df['Cabin'].isnull()

df['Fare']=df['Fare'].fillna(df['Fare'].median())

df['Embarked']=df['Embarked'].fillna(df['Embarked'].mode()[0])
df.isnull().sum()
plt.figure(figsize=(12,4))

plt.subplot(1,2,1)

sns.distplot(df[df.Survived==1].Age, color='green', kde=False)

sns.distplot(df[df.Survived==0].Age, color='orange', kde=False)

plt.subplot(1,2,2)

plt.xlim(0,100)

sns.distplot(df[df.Survived==1].Fare, color='green', kde=False)

sns.distplot(df[df.Survived==0].Fare, color='orange', kde=False)

plt.show()

df['family']=df['SibSp']+df['Parch']

df['family'].head()
df.head()
df.drop(['Cabin','Ticket','PassengerId','Name','SibSp','Parch'],axis=1,inplace=True)
df.head()
df.isnull().sum()
train_data=df.iloc[:891,:]

test_data=df.iloc[891:,:]

train_data=pd.get_dummies(train_data,columns=['Pclass','Sex','Embarked'],prefix=['Pclass','Sex','Embarked'])

test_data=pd.get_dummies(test_data,columns=['Pclass','Sex','Embarked'],prefix=['Pclass','Sex','Embarked'])
xtrain=train_data.drop(['Survived'],axis=1)

ytrain=train_data['Survived']

xtest=test_data.drop(['Survived'],axis=1)



xtrain.drop(['Title'],axis=1,inplace=True)

xtest.drop(['Title'],axis=1,inplace=True)
logreg = LogisticRegression()

logreg.fit(xtrain, ytrain)

Y_pred = logreg.predict(xtest)

acc_log = round(logreg.score(xtrain, ytrain) * 100, 2)

acc_log



svc = SVC()

svc.fit(xtrain, ytrain)

Y_pred = svc.predict(xtest)

acc_svc = round(svc.score(xtrain, ytrain) * 100, 2)

acc_svc



random_forest = RandomForestClassifier(n_estimators=10)

random_forest.fit(xtrain, ytrain)

Y_pred = random_forest.predict(xtest)

random_forest.score(xtrain, ytrain)

acc_random_forest = round(random_forest.score(xtrain, ytrain) * 100, 2)

acc_random_forest

linear_svc = LinearSVC()

linear_svc.fit(xtrain, ytrain)

Y_pred = linear_svc.predict(xtest)

acc_linear_svc = round(linear_svc.score(xtrain, ytrain) * 100, 2)

acc_linear_svc
decision_tree = DecisionTreeClassifier(max_depth=10)

decision_tree.fit(xtrain, ytrain)

Y_pred = decision_tree.predict(xtest)

acc_decision_tree = round(decision_tree.score(xtrain, ytrain) * 100, 2)

acc_decision_tree