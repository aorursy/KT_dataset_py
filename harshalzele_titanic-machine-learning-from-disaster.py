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
import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')
train=pd.read_csv("../input/train.csv")

test=pd.read_csv("../input/test.csv")
train.head()
train.describe(include="all")
train.info()
round(100*(train.isnull().sum()/len(train.index)),2)
train.isnull().sum()
print("Female", train["Survived"][train["Sex"] == 'female'].value_counts(normalize = True)[1]*100)

print("Male", train["Survived"][train["Sex"] == 'male'].value_counts(normalize = True)[1]*100)

sns.barplot(x='Sex', y='Survived', data=train) ## Analysing the Male survived versus Female
print("1 - Upper Class", train['Survived'][train['Pclass'] == 1].value_counts(normalize = True)[1]*100)

print("2 - Middle Class", train['Survived'][train['Pclass'] == 2].value_counts(normalize = True)[1]*100)

print("3 - Lower Class", train['Survived'][train['Pclass'] == 3].value_counts(normalize = True)[1]*100)

sns.barplot(x='Pclass',y='Survived',data=train) # Checking class wise survived.
sns.barplot(x='SibSp',y='Survived',data=train)

print("SibSp=0", train['Survived'][train['SibSp']==0].value_counts(normalize=True)[1]*100)

print("SibSp=1", train['Survived'][train['SibSp']==1].value_counts(normalize=True)[1]*100)

print("SibSp=2", train['Survived'][train['SibSp']==2].value_counts(normalize=True)[1]*100)

print("SibSp=3", train['Survived'][train['SibSp']==3].value_counts(normalize=True)[1]*100)

print("SibSp=4", train['Survived'][train['SibSp']==4].value_counts(normalize=True)[1]*100)
sns.barplot(x='Parch',y='Survived',data=train)

print("Parch = 0", round(train['Survived'][train['Parch']==0].value_counts(normalize=True)[1]*100,2), "%")

print("Parch = 1", round(train['Survived'][train['Parch']==1].value_counts(normalize=True)[1]*100,2), "%")

print("Parch = 2", round(train['Survived'][train['Parch']==2].value_counts(normalize=True)[1]*100,2), "%")

print("Parch = 3", round(train['Survived'][train['Parch']==3].value_counts(normalize=True)[1]*100,2), "%")

#print("Parch = 4", round(train['Survived'][train['Parch']==4].value_counts(normalize=True)[1]*100,2), "%")

print("Parch = 5", round(train['Survived'][train['Parch']==5].value_counts(normalize=True)[1]*100,2), "%")
#sort the ages into logical categories

train["Age"] = train["Age"].fillna(-0.5)

test["Age"] = test["Age"].fillna(-0.5)

bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf]

labels = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']

train['AgeGroup'] = pd.cut(train["Age"], bins, labels = labels)

test['AgeGroup'] = pd.cut(test["Age"], bins, labels = labels)



#draw a bar plot of Age vs. survival

sns.barplot(x="AgeGroup", y="Survived", data=train)

plt.show()
train['cabinbool']=train['Cabin'].notnull().astype(int)

test['cabinbool']=test['Cabin'].notnull().astype(int)
print("Cabin alloted = ", round(train['Survived'][train['cabinbool']==1].value_counts(normalize=True)[1]*100,2), "%")

print("Cabin not alloted= ", round(train['Survived'][train['cabinbool']==0].value_counts(normalize=True)[1]*100,2), "%")
sns.barplot(x='cabinbool',y='Survived',data=train)
# Dropping this because it has 77% null data

train=train.drop(['Cabin'],axis=1)

test=test.drop(['Cabin'],axis=1)
#we can also drop the Ticket feature since it's unlikely to yield any useful information

train = train.drop(['Ticket'], axis = 1)

test = test.drop(['Ticket'], axis = 1)
print ('Numer of People embarked from Southampton: ', train[train['Embarked']=='S'].shape[0])

print ('Numer of People embarked from Cherbourg: ', train[train['Embarked']=='C'].shape[0])

print ('Numer of People embarked from Queenstown: ', train[train['Embarked']=='Q'].shape[0])
#0.22% values are miising. So imputing it with popular one "S"

train['Embarked']=train['Embarked'].fillna(value='S')
#create a combined group of both datasets

combine = [train, test]



#extract a title for each Name in the train and test datasets

for dataset in combine:

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)



pd.crosstab(train['Title'], train['Sex'])
for dataset in combine:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Capt', 'Col',

    'Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Don'], 'Rare')    

    dataset['Title'] = dataset['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')



train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
#map each of the title groups to a numerical value

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Royal": 5, "Rare": 6}

for dataset in combine:

    dataset['Title'] = dataset['Title'].map(title_mapping)

    dataset['Title'] = dataset['Title'].fillna(0)



train.head()
# fill missing age with mode age group for each title

mr_age = train[train["Title"] == 1]["AgeGroup"].mode() #Young Adult

miss_age = train[train["Title"] == 2]["AgeGroup"].mode() #Student

mrs_age = train[train["Title"] == 3]["AgeGroup"].mode() #Adult

master_age = train[train["Title"] == 4]["AgeGroup"].mode() #Baby

royal_age = train[train["Title"] == 5]["AgeGroup"].mode() #Adult

rare_age = train[train["Title"] == 6]["AgeGroup"].mode() #Adult



age_title_mapping = {1: "Young Adult", 2: "Student", 3: "Adult", 4: "Baby", 5: "Adult", 6: "Adult"}



for x in range(len(train["AgeGroup"])):

    if train["AgeGroup"][x] == "Unknown":

        train["AgeGroup"][x] = age_title_mapping[train["Title"][x]]

        

for x in range(len(test["AgeGroup"])):

    if test["AgeGroup"][x] == "Unknown":

        test["AgeGroup"][x] = age_title_mapping[test["Title"][x]]
#map each Age value to a numerical value

age_mapping = {'Baby': 1, 'Child': 2, 'Teenager': 3, 'Student': 4, 'Young Adult': 5, 'Adult': 6, 'Senior': 7}

train['AgeGroup'] = train['AgeGroup'].map(age_mapping)

test['AgeGroup'] = test['AgeGroup'].map(age_mapping)



train.head()
train= train.drop(['Age'], axis=1)

test= test.drop(['Age'], axis=1)
age_map={"male":0, "female":1}

train['Sex']=train['Sex'].map(age_map)

test['Sex']=test['Sex'].map(age_map)
embarked_map={"S":0, "C":1,"Q":2}

train['Embarked']=train['Embarked'].map(embarked_map)

test['Embarked']=test['Embarked'].map(embarked_map)
#fill in missing Fare value in test set based on mean fare for that Pclass 

for x in range(len(test["Fare"])):

    if pd.isnull(test["Fare"][x]):

        pclass = test["Pclass"][x] #Pclass = 3

        test["Fare"][x] = round(train[train["Pclass"] == pclass]["Fare"].mean(), 4)



#map Fare values into groups of numerical values

train['FareBand'] = pd.qcut(train['Fare'], 4, labels = [1, 2, 3, 4])

test['FareBand'] = pd.qcut(test['Fare'], 4, labels = [1, 2, 3, 4])
train = train.drop(['Fare'], axis = 1)

test = test.drop(['Fare'], axis = 1)



train = train.drop(['Name'], axis = 1)

test = test.drop(['Name'], axis = 1)
train.head(10)
test.head(10)
train.info()
from sklearn.model_selection import train_test_split



predictors = train.drop(['Survived', 'PassengerId'], axis=1)

target = train["Survived"]

x_train, x_val, y_train, y_val = train_test_split(predictors, target, test_size = 0.22, random_state = 0)
# Logistic Regression

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score



logreg = LogisticRegression()

logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_val)

acc_logreg = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_logreg)
#set ids as PassengerId and predict survival 

ids = test['PassengerId']

predictions = logreg.predict(test.drop('PassengerId', axis=1))



#set the output as a dataframe and convert to csv file named submission.csv

output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })

output.to_csv('submission.csv', index=False)