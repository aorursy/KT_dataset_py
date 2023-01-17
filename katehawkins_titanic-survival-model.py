# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#data analysis libraries 

import numpy as np

import pandas as pd



#visualization libraries

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



#import train and test CSV files

train = pd.read_csv("/kaggle/input/titanic/train.csv")

test = pd.read_csv("/kaggle/input/titanic/test.csv")



#take a look at the training data

train.describe(include="all")
print(train.columns)
train.sample(5)
train.dtypes
print("No. Null Values in Training Data:")

print(train.isnull().sum())

#missing data for Age (177 records), Cabin (687 records), and Embarked (2)
#draw a bar plot of Survival by Sex

sns.barplot(x="Sex", y="Survived", data=train)



#print percentages of Females vs. Males that survive

print("Percentage of females who survived:", train["Survived"][train["Sex"] == 'female'].value_counts(normalize = True)[1]*100)



print("Percentage of males who survived:", train["Survived"][train["Sex"] == 'male'].value_counts(normalize = True)[1]*100)
#draw a bar plot of Survival by Pclass

sns.barplot(x='Pclass', y='Survived', data=train)



#print percentage of 1/2/3 Pclass survival rate

print("Percentage of 1st class passengers that survived:", train['Survived'][train['Pclass']==1].value_counts(normalize = True)[1]*100)

print("Percentage of 2nd class passengers that survived:", train['Survived'][train['Pclass']==2].value_counts(normalize = True)[1]*100)

print("Percentage of 3rd class passengers that survived:", train['Survived'][train['Pclass']==3].value_counts(normalize = True)[1]*100)

#draw a bar plot of Survival by Number of Siblings/Spouses

sns.barplot(x='SibSp', y='Survived', data=train)



#print percentage of survival rate by number of siblings/spouses

print("Percentage survival where SibSp = 0:", train['Survived'][train['SibSp']==0].value_counts(normalize = True)[1]*100)

print("Percentage survival where SibSp = 1:", train['Survived'][train['SibSp']==1].value_counts(normalize = True)[1]*100)

print("Percentage survival where SibSp = 2:", train['Survived'][train['SibSp']==2].value_counts(normalize = True)[1]*100)

print("Percentage survival where SibSp = 3:", train['Survived'][train['SibSp']==3].value_counts(normalize = True)[1]*100)

print("Percentage survival where SibSp = 4:", train['Survived'][train['SibSp']==4].value_counts(normalize = True)[1]*100)

#draw a bar plot of Survival by No. Parents/Children onboard

sns.barplot(x='Parch', y='Survived', data=train)



#print percentage survival by no. of parents/children onboard

print("Percentage survival where Parch = 0:", train['Survived'][train['Parch']==0].value_counts(normalize = True)[1]*100)

print("Percentage survival where Parch = 1:", train['Survived'][train['Parch']==1].value_counts(normalize = True)[1]*100)

print("Percentage survival where Parch = 2:", train['Survived'][train['Parch']==2].value_counts(normalize = True)[1]*100)

print("Percentage survival where Parch = 3:", train['Survived'][train['Parch']==3].value_counts(normalize = True)[1]*100)

print("Percentage survival where Parch = 5:", train['Survived'][train['Parch']==5].value_counts(normalize = True)[1]*100)
#sort the ages into logical categories

train["Age"] = train["Age"].fillna(-0.5)

test["Age"] = test["Age"].fillna(-0.5)



#use bins to categorize continuous data into discrete groups 

bins = [-1, 0, 5, 12, 18, 35, 60, np.inf]

labels = ['Unknown', 'Baby', 'Child', 'Teen', 'Young Adult', 'Adult', 'Senior']



#creating a new column with Age Group (as defined in the bins above)

train['AgeGroup'] = pd.cut(train["Age"], bins, labels = labels)

test['AgeGroup'] = pd.cut(test["Age"], bins, labels = labels)



#draw a bar plot of Age vs. survival

sns.barplot(x="AgeGroup", y="Survived", data=train)



#print survival rate for different age categories

print("Survival Rate for Babies: ", train['Survived'][train['AgeGroup']=='Baby'].value_counts(normalize = True)[1]*100)

print("Survival Rate for Children: ", train['Survived'][train['AgeGroup']=='Child'].value_counts(normalize = True)[1]*100)

print("Survival Rate for Teenagers: ", train['Survived'][train['AgeGroup']=='Teen'].value_counts(normalize = True)[1]*100)

print("Survival Rate for Young Adults: ", train['Survived'][train['AgeGroup']=='Young Adult'].value_counts(normalize = True)[1]*100)

print("Survival Rate for Adults: ", train['Survived'][train['AgeGroup']=='Adult'].value_counts(normalize = True)[1]*100)

print("Survival Rate for Seniors: ", train['Survived'][train['AgeGroup']=='Senior'].value_counts(normalize = True)[1]*100)
#Cabin feature has lots of missing values, so column could be seen as not worth keeping

#HOWEVER, presence of cabin number can be indicative of higher socioeconomic class 



#creating a new column with a 1/0 indicating whether a cabin number is present for that record 

train["CabinBool"] = (train["Cabin"].notnull().astype('int'))

test["CabinBool"] = (test["Cabin"].notnull().astype('int'))



#draw a bar plot of CabinBool vs. survival

sns.barplot(x="CabinBool", y="Survived", data=train)



#calculate percentages of CabinBool vs. survived

print("Percentage of CabinBool = 1 who survived:", train["Survived"][train["CabinBool"] == 1].value_counts(normalize = True)[1]*100)

print("Percentage of CabinBool = 0 who survived:", train["Survived"][train["CabinBool"] == 0].value_counts(normalize = True)[1]*100)
#Editing the Data



print(train.columns)



#can now drop the following features as they provide no information 

train = train.drop(['Cabin'], axis = 1)

test = test.drop(['Cabin'], axis = 1)

train = train.drop(['Ticket'], axis= 1)

test = test.drop(['Ticket'], axis = 1)



#Embarked Feature



print("Number of people embarking in Southampton (S): ",train[train["Embarked"] == "S"].shape[0])

print("Number of people embarking in Cherbourg (C):   ", train[train["Embarked"] == "C"].shape[0])

print("Number of people embarking in Queenstown (Q):   ", train[train['Embarked'] == "Q"].shape[0])

print("No. Null Values in Training Data:", train['Embarked'].isnull().sum())
#can see the majority of people embarked at Southampton (modal value), so can replace missing values with Southampton

train = train.fillna({"Embarked" : "S"})

print("No. Null Values in Training Data: ", train["Embarked"].isnull().sum())
#Age feature: we have lots of missing values, so want to be more refined than filling in missing values with the overall average



#create a combined group of both datasets

combined_data = [train, test]



#extract a title for each Name in the train and test datasets

for dataset in combined_data:

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)



pd.crosstab(train['Title'], train['Sex'])
#replace various titles with more common names

for dataset in combined_data:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Capt', 'Col',

    'Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare') 

    dataset['Title'] = dataset['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')



print(train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())

master_age = train[train['Title'] == 'Master']['Age'].mean()

miss_age = train[train['Title'] == 'Miss']['Age'].mean()

mr_age = train[train['Title'] == 'Mr']['Age'].mean()

mrs_age = train[train['Title'] == 'Mrs']['Age'].mean()

rare_age = train[train['Title'] == 'Rare']['Age'].mean()

royal_age = train[train['Title'] == 'Royal']['Age'].mean()



print(master_age, miss_age, mr_age, mrs_age, rare_age, royal_age)

print(train[train['Title']=='Miss']['Age'].sample(5))
#for each Title, where a record has a missing value for Age (which I earlier assigned a value of -0.5), I am giving it the mean age for that Title group

for dataset in combined_data:

    dataset[dataset['Title']=='Master'] = dataset[dataset['Title']=='Master'].replace(-0.5, master_age)

    dataset[dataset['Title']=='Miss'] = dataset[dataset['Title']=='Miss'].replace(-0.5, miss_age)

    dataset[dataset['Title']=='Mr'] = dataset[dataset['Title']=='Mr'].replace(-0.5, mr_age)

    dataset[dataset['Title']=='Mrs'] = dataset[dataset['Title']=='Mrs'].replace(-0.5, mrs_age)

    dataset[dataset['Title']=='Rare'] = dataset[dataset['Title']=='Rare'].replace(-0.5, rare_age)

    dataset[dataset['Title']=='Royal'] = dataset[dataset['Title']=='Royal'].replace(-0.5, royal_age)



#check to see if this has solved the problem of missing values

print(train.isnull().sum())

print(test.isnull().sum())
train.sample(10)
#recategorising the bins so that the records originally missing an age value (now given the average age for their title) can be put into an age group

bins = [0, 5, 12, 18, 35, 60, np.inf]

labels = ['Baby', 'Child', 'Teen', 'Young Adult', 'Adult', 'Senior']



train['AgeGroup'] = pd.cut(train["Age"], bins, labels = labels)

test['AgeGroup'] = pd.cut(test["Age"], bins, labels = labels)



#check to see if all records have been grouped and given a general title and age

print(train[['Title', 'AgeGroup']].groupby(['Title'], as_index=False).count())

print(test[['Title', 'AgeGroup']].groupby(['Title'], as_index=False).count())



#can now drop Age feature as we have categorised all records into an age group

train = train.drop(['Age'], axis=1)

test = test.drop(['Age'], axis=1)



#can now drop name feature as we have extracted the useful information from it i.e. title

train = train.drop(['Name'], axis=1)

test = test.drop(['Name'], axis=1)

#Fare Feature

print(test.isnull().sum())
#1 missing value for Fare in test data, will replace this with the mean fare for the dataset

test["Fare"] = test["Fare"].fillna(test['Fare'].mean())

print(test.isnull().sum())
#testing the data

from sklearn.ensemble import RandomForestClassifier



y = train['Survived']



features = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Fare', 'Embarked', 'AgeGroup', 'CabinBool', 'Title']

X_train = pd.get_dummies(train[features])

X_test = pd.get_dummies(test[features])



print(X_train)

print(X_test)



#X_test has one fewer column than X_train becuase there were no records where Title == Royal, hence no dummy was created for this instance in the test data

#no. input features != no. model features --> model wouldn't run --> add in a feature to test data with a dummy of 0 for Title == Royal for all records 

X_test['Title_Royal'] = 0



rf_model = RandomForestClassifier()

rf_model.fit(X_train, y)

predictions = rf_model.predict(X_test)





output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")
#testing accuracy of model by splitting the training data

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split



predictors = pd.get_dummies(train.drop(['Survived', 'PassengerId'], axis=1))

target = pd.get_dummies(train["Survived"])

x_train, x_val, y_train, y_val = train_test_split(predictors, target, test_size = len(test)/len(train), random_state = 0)





randomforest = RandomForestClassifier()

randomforest.fit(x_train, y_train)

y_pred = randomforest.predict(x_val)

acc_randomforest = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_randomforest)