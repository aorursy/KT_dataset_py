import numpy as np 

import pandas as pd 



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
#import train and test CSV files

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")



#take a look at the training data

train.head()
train.describe(include="all")
print(train.columns)
#check for any other unusable values

print(pd.isnull(train).sum())
#print percentages of females vs. males that survive

print("Percentage of females who survived:", train["Survived"][train["Sex"] == 'female'].value_counts(normalize = True)[1]*100)



print("Percentage of males who survived:", train["Survived"][train["Sex"] == 'male'].value_counts(normalize = True)[1]*100)

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
train["CabinBool"] = (train["Cabin"].notnull().astype('int'))

test["CabinBool"] = (test["Cabin"].notnull().astype('int'))



#calculate percentages of CabinBool vs. survived

print("Percentage of CabinBool = 1 who survived:", train["Survived"][train["CabinBool"] == 1].value_counts(normalize = True)[1]*100)



print("Percentage of CabinBool = 0 who survived:", train["Survived"][train["CabinBool"] == 0].value_counts(normalize = True)[1]*100)

#draw a bar plot of CabinBool vs. survival

sns.barplot(x="CabinBool", y="Survived", data=train)

plt.show()
train = train.drop(['Cabin'], axis = 1)

test = test.drop(['Cabin'], axis = 1)

#we can also drop the Ticket feature since it's unlikely to yield any useful information

train = train.drop(['Ticket'], axis = 1)

test = test.drop(['Ticket'], axis = 1)
#now we need to fill in the missing values in the Embarked feature

print("Number of people embarking in Southampton (S):")

southampton = train[train["Embarked"] == "S"].shape[0]

print(southampton)



print("Number of people embarking in Cherbourg (C):")

cherbourg = train[train["Embarked"] == "C"].shape[0]

print(cherbourg)



print("Number of people embarking in Queenstown (Q):")

queenstown = train[train["Embarked"] == "Q"].shape[0]

print(queenstown)



train = train.fillna({"Embarked": "S"})
#create a combined group of both datasets

combine = [train, test]



#extract a title for each Name in the train and test datasets

for dataset in combine:

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)



pd.crosstab(train['Title'], train['Sex'])
for dataset in combine:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Capt', 'Col',

    'Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')

    

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



#I tried to get this code to work with using .map(), but couldn't.

#I've put down a less elegant, temporary solution for now.

#train = train.fillna({"Age": train["Title"].map(age_title_mapping)})

#test = test.fillna({"Age": test["Title"].map(age_title_mapping)})



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
#dropping the Age feature for now, might change

train = train.drop(['Age'], axis = 1)

test = test.drop(['Age'], axis = 1)



#drop the name feature since it contains no more useful information.

train = train.drop(['Name'], axis = 1)

test = test.drop(['Name'], axis = 1)



#map each Sex value to a numerical value

sex_mapping = {"male": 0, "female": 1}

train['Sex'] = train['Sex'].map(sex_mapping)

test['Sex'] = test['Sex'].map(sex_mapping)



train.head()





#map each Embarked value to a numerical value

embarked_mapping = {"S": 1, "C": 2, "Q": 3}

train['Embarked'] = train['Embarked'].map(embarked_mapping)

test['Embarked'] = test['Embarked'].map(embarked_mapping)



train.head()



#fill in missing Fare value in test set based on mean fare for that Pclass 

for x in range(len(test["Fare"])):

    if pd.isnull(test["Fare"][x]):

        pclass = test["Pclass"][x] #Pclass = 3

        test["Fare"][x] = round(train[train["Pclass"] == pclass]["Fare"].mean(), 4)

        

#map Fare values into groups of numerical values

train['FareBand'] = pd.qcut(train['Fare'], 4, labels = [1, 2, 3, 4])

test['FareBand'] = pd.qcut(test['Fare'], 4, labels = [1, 2, 3, 4])



#drop Fare values

train = train.drop(['Fare'], axis = 1)

test = test.drop(['Fare'], axis = 1)



train.head()
from sklearn.model_selection import train_test_split



predictors = train.drop(['Survived', 'PassengerId'], axis=1)

target = train["Survived"]

x_train, x_val, y_train, y_val = train_test_split(predictors, target, test_size = 0.22, random_state = 0)
# Support Vector Machines

from sklearn.svm import SVC

from sklearn.metrics import accuracy_score

svc = SVC()

svc.fit(x_train, y_train)

y_pred = svc.predict(x_val)

acc_svc = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_svc)
#set ids as PassengerId and predict survival 

ids = test['PassengerId']

predictions = svc.predict(test.drop('PassengerId', axis=1))



#set the output as a dataframe and convert to csv file named submission.csv

output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })

output.to_csv('submission.csv', index=False)