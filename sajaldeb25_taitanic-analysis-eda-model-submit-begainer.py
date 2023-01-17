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

        os.path.join(dirname, filename)



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import random as rnd



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier
train_df = pd.read_csv('../input/titanic/train.csv')

test_df = pd.read_csv('../input/titanic/test.csv')
train_df.head(10)
test_df.tail(10)
submission = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": 1

    })

submission.to_csv('submission.csv', index=False)
test_df2 = pd.read_csv('../input/titanic/test.csv')
test_df2.head()
test_df['sur'] = 0
test_df.head()
test_df = test_df.drop(['sur'], axis=1)
test_df.head()
test_df2['sur'] = 0
test_df2.head()
# Mark all female had been survived thats why sur=1

test_df2.loc[test_df2.Sex == 'female', 'sur'] = 1
test_df2.head()
submission = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": test_df2["sur"]

    })
submission.to_csv('submission.csv', index=False)
# fare less than 20 and who belong Pclass = 3 and who was female, mark then and died.

test_df2.loc[(test_df2.Fare > 20) & (test_df2['Pclass'] == 3) & (test_df['Sex']== 'female') , 'sur'] = 0
test_df2.head()
# Update submission file 

submission = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": test_df2["sur"]

    })
submission.to_csv('submission.csv', index=False)
train_df = pd.read_csv('../input/titanic/train.csv')

test_df = pd.read_csv('../input/titanic/test.csv')
train_df.shape
test_df.shape
train_df.info()

# observe some missing data in Age, Cabin, Embarked
test_df.info()
train_df.isnull().sum()
test_df.isnull().sum()
#define a function, so that we can make bar chart for every feature. 

def barchart(feature):

    g = sns.barplot(x=feature,y="Survived",data=train_df)

    g = g.set_ylabel("Survival Probability")
# For sex feature. And see most of Feamale passenger had beed survived.

barchart('Sex')
barchart('Pclass')
barchart('SibSp')
barchart('Parch')
barchart('Embarked')
# Marged train and test data set.

train_test_data = [train_df, test_df]

for dataset in train_test_data:

    dataset['Title'] = dataset['Name'].str.extract('([A-Za-z]+)\.', expand = False)
train_df['Title'].value_counts()
test_df['Title'].value_counts()
# extract example 

s2 = pd.Series(['a_b, Dr. c', 'c_d001. e', 'ADR, Mr1. Sajal', 'f_g8.h','as, Miss. Angel'])

s2.str.extract('([A-Za-z]+[0-9]+)\.')
s2.str.extract('([A-Za-z]+)\.')
#Mapping the unnecessary title with 0,1,2,3

title_mapping = {"Mr": 0,"Miss": 1,"Mrs": 2,"Master": 3,"Dr": 3,"Rev": 3,"Mlle": 3,"Countess": 3,"Ms": 3,"Lady": 3,"Jonkheer": 3,"Don": 3,"Dona": 3,"Mme": 3,"Capt": 3,"Sir": 3,"Col":3,"Major":3 }



for dataset in train_test_data:

    dataset['Title']  = dataset['Title'].map(title_mapping)
test_df['Title'].value_counts()
train_df['Title'].value_counts()
train_df.info()
# Delete unnecessary feature from dataset

train_df.drop('Name',axis=1,inplace=True)

test_df.drop('Name',axis=1,inplace=True)
test_df.head()
#Mapping Male and Female in number 

sex_mapping = {"male": 0,"female": 1 }



for dataset in train_test_data:

    dataset['Sex']  = dataset['Sex'].map(sex_mapping)
test_df.head()
barchart('Sex')
# FIll missing age with measian age of passengers 

train_df["Age"].fillna(train_df.groupby("Title")["Age"].transform("median"), inplace=True)

test_df["Age"].fillna(test_df.groupby("Title")["Age"].transform("median"), inplace=True)
train_df.head()
# See -> Age are now not NULL

train_df.info()
test_df.info()
# For better understanding we make some chart for age 



facet = sns.FacetGrid(train_df,hue="Survived", aspect=4)

facet.map(sns.kdeplot, 'Age', shade=True)

facet.set(xlim=(0,train_df['Age'].max()))

facet.add_legend()



plt.show()
facet = sns.FacetGrid(train_df,hue="Survived", aspect=4)

facet.map(sns.kdeplot, 'Age', shade=True)

facet.set(xlim=(0,train_df['Age'].max()))

facet.add_legend()



plt.xlim(0,20)

# plt.show()
facet = sns.FacetGrid(train_df,hue="Survived", aspect=4)

facet.map(sns.kdeplot, 'Age', shade=True)

facet.set(xlim=(0,train_df['Age'].max()))

facet.add_legend()



plt.xlim(20,30)
facet = sns.FacetGrid(train_df,hue="Survived", aspect=4)

facet.map(sns.kdeplot, 'Age', shade=True)

facet.set(xlim=(0,train_df['Age'].max()))

facet.add_legend()



plt.xlim(30,40)
facet = sns.FacetGrid(train_df,hue="Survived", aspect=4)

facet.map(sns.kdeplot, 'Age', shade=True)

facet.set(xlim=(0,train_df['Age'].max()))

facet.add_legend()



plt.xlim(40,80)
# Make category for age in five as child=0, young=1, adult=2, mid_age=3, senior=4

for dataset in train_test_data:

    dataset.loc[dataset['Age'] <=16, 'Age'] = 0,

    dataset.loc[(dataset['Age'] >16) & (dataset['Age'] <=26), 'Age'] = 1,

    dataset.loc[(dataset['Age'] >26) & (dataset['Age'] <=36), 'Age'] = 2,

    dataset.loc[(dataset['Age'] >36) & (dataset['Age'] <=62), 'Age'] = 3,

    dataset.loc[dataset['Age'] >62, 'Age'] = 4
train_df.head()
barchart('Age')
# filling missing value of Embarked

for dataset in train_test_data:

    dataset['Embarked']  = dataset['Embarked'].fillna('S')
train_df.info()
embarked_map = {"S":0, "C":1, "Q":2}

for dataset in train_test_data:

    dataset['Embarked']  = dataset['Embarked'].map(embarked_map)
train_df.head()
test_df.info()
# FIll missing Fare with measian age of passengers 

train_df["Fare"].fillna(train_df.groupby("Pclass")["Fare"].transform("median"), inplace=True)

test_df["Fare"].fillna(test_df.groupby("Pclass")["Fare"].transform("median"), inplace=True)

test_df.info()
# For better understanding we make some chart for Fare



facet = sns.FacetGrid(train_df,hue="Survived", aspect=4)

facet.map(sns.kdeplot, 'Fare', shade=True)

facet.set(xlim=(0,train_df['Fare'].max()))

facet.add_legend()



plt.show()
facet = sns.FacetGrid(train_df,hue="Survived", aspect=4)

facet.map(sns.kdeplot, 'Fare', shade=True)

facet.set(xlim=(0,train_df['Fare'].max()))

facet.add_legend()

plt.xlim(0,20)
facet = sns.FacetGrid(train_df,hue="Survived", aspect=4)

facet.map(sns.kdeplot, 'Fare', shade=True)

facet.set(xlim=(0,train_df['Fare'].max()))

facet.add_legend()

plt.xlim(20,30)
facet = sns.FacetGrid(train_df,hue="Survived", aspect=4)

facet.map(sns.kdeplot, 'Fare', shade=True)

facet.set(xlim=(0,train_df['Fare'].max()))

facet.add_legend()

plt.xlim(30,100)
# Make category for FARE in four 

for dataset in train_test_data:

    dataset.loc[dataset['Fare'] <=7.5, 'Fare'] = 0,

    dataset.loc[(dataset['Fare'] >7.5) & (dataset['Fare'] <=15), 'Fare'] = 1,

    dataset.loc[(dataset['Fare'] >15) & (dataset['Fare'] <=30), 'Fare'] = 2,

    dataset.loc[(dataset['Fare'] >30) & (dataset['Fare'] <=100), 'Fare'] = 3,

    dataset.loc[dataset['Fare'] >100, 'Fare'] = 4
train_df.head(20)
# work with Cabin 

train_df.Cabin.value_counts()
for dataset in train_test_data:

    dataset['Cabin'] = dataset['Cabin'].str[:1]
train_df.Cabin.value_counts()
pclass1 = train_df[train_df['Pclass'] == 1]['Cabin'].value_counts()

pclass2 = train_df[train_df['Pclass'] == 2]['Cabin'].value_counts()

pclass3 = train_df[train_df['Pclass'] == 3]['Cabin'].value_counts()

df = pd.DataFrame([pclass1,pclass2,pclass3])

df.index = ['1st class','2nd class','3rd class']

df.plot(kind='bar',stacked=True, figsize=(10,5))
# Cabin Mapping 

cabin_mapping = {"A":0,"B":0.4,"C":0.8,"D":1.2,"E":1.6,"F":2,"G":2.4,"T":2.8}

for dataset in train_test_data:

    dataset['Cabin']  = dataset['Cabin'].map(cabin_mapping)
# filling missing Fare with median fare for each Pclass

train_df["Cabin"].fillna(train_df.groupby("Pclass")["Cabin"].transform("median"), inplace=True)

test_df["Cabin"].fillna(test_df.groupby("Pclass")["Cabin"].transform("median"), inplace=True)
train_df.info()
test_df.info()
# For family size

train_df["FamilySize"] = train_df["SibSp"]+ train_df["Parch"]+1
test_df["FamilySize"] = test_df["SibSp"]+ test_df["Parch"]+ 1
facet = sns.FacetGrid(train_df,hue="Survived", aspect=4)

facet.map(sns.kdeplot, 'FamilySize', shade=True)

facet.set(xlim=(0,train_df['FamilySize'].max()))

facet.add_legend()

plt.xlim(0)
# Family Mapping 

family_mapping = {1:0,2:0.4,3:0.8,4:1.2,5:1.6,6:2,7:2.4,8:2.8,9:3.2,10:3.6,11:4}

for dataset in train_test_data:

    dataset['FamilySize']  = dataset['FamilySize'].map(family_mapping)
train_df.head()
# Dropping the unnecessary feature

frdp = ['Ticket','SibSp','Parch']

train_df = train_df.drop(frdp, axis=1)

test_df = test_df.drop(frdp, axis=1)

train_df = train_df.drop(['PassengerId'], axis=1)
train_df.head()
# Importing Classifier Modules

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier
# cross validatin with KFold

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

K_fold = KFold(n_splits=10,shuffle=True,random_state =0)
clf = DecisionTreeClassifier()

scoring = 'accuracy'

x = train_df.drop('Survived',axis=1)

y = train_df['Survived']

score = cross_val_score(clf ,x ,y , cv=K_fold, n_jobs=1, scoring=scoring)

print(score)
#decision tree Score

round(np.mean(score)*100,2)
clf = RandomForestClassifier(n_estimators=13)

scoring = 'accuracy'

score = cross_val_score(clf ,x ,y , cv=K_fold, n_jobs=1, scoring=scoring)

print(score)
#Random Forest Score

round(np.mean(score)*100,2)
clf = RandomForestClassifier(n_estimators=13)

clf.fit(x, y)



test_data = test_df.drop("PassengerId", axis=1).copy()

test_data.info()
prediction = clf.predict(test_data)
# Update submission file 

submission = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": prediction

    })

submission.to_csv('submission.csv',index=False)
submission = pd.read_csv('submission.csv')

submission.head()