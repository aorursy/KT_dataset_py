# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
#ignore warnings
import warnings
warnings.filterwarnings('ignore')
#import train and test CSV files

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

#take a look at the training data
train.describe(include="all")
# get a list of the features within the dataset
print(train.columns)
# see a sample of the dataset to get an idea of the variables
train.sample(5)
print(pd.isnull(train).sum())
#draw a bar plot of suvival by sex
sns.barplot(x="Sex", y = "Survived", data=train)

#print percentages of females vs. males that survive
print("Percentage of females who survived:", train["Survived"][train["Sex"] =='female'].value_counts(normalize = True)[1]*100)
print("Percentage of males who survived:", train["Survived"][train["Sex"] == 'male'].value_counts(normalize = True)[1]*100)
#Pclass Feature
#draw a bar plot of survival by Pclass
sns.barplot(x="Pclass", y="Survived", data=train)

#print percentage of people by Pclass that survived
print("Percentage of Pclass = 1 who survived:", train["Survived"][train["Pclass"] == 1].value_counts(normalize = True)[1]*100)
print("Percentage of Pclass = 2 who survived:", train["Survived"][train["Pclass"] == 2].value_counts(normalize = True)[1]*100)
print("percentage of Pclass = 3 who survived:", train["Survived"][train["Pclass"] == 3].calue_counts(normalize = True)[1]*100)
#SibSP Feature
#draw a bar plot for SibSp vs. Survival
sns.barplot(x="SibSp", y="Survived", data=train)

#I Won't be printing indivudual percent values for all of these. 
print("Percentage of Sibsp = 0 who survived:", train["Survived"][train["SibSp"] == 0].value_counts(normalize = True)[1]*100)

print("Percentage of SibSP = 1 who survived:", train["Survived"][train["SibSp"] == 1].value_counts(normalize = True)[1]*100)

print("Percentage of SibSp = 2 who survived:", train["Survived"][train["SibSp"] == 2].value_counts(normalize = True)[1]*100)
#Parch Feature
#draw a bar plot for parch vs. survival 
sns.barplot(x="Parch", y="Survived", data=train)
plt.show()
#Age Feature
# sort the ages into logical categories
train["Age"] = train["Age"].fillna(-0.5)
test["Age"] =test["Age"].fillna(-0.5)
bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf]
labels = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
train['AgeGroup'] = pd.cut(train["Age"], bins, labels =labels)
test['AgeGroup'] = pd.cut(test["Age"], bins, labels = labels)

#draw a bar plot of Age vs. survival
sns.barplot(x="AgeGroup", y="Survived", data=train)
plt.show
#Cabin Feature
# I think the idea here is that people with recorded cabin numbers are of higher socioeconomic class, and thus more likely to survive.

train["CabinBool"] = (train["Cabin"].notnull().astype('int'))
test["CabinBool"] = (test["Cabin"].notnull().astype('int'))

#calculate percentages of CabinBool vs. survived
print ("Percentage of CabinBool = 1 who survived:", train ["Survived"][train["CabinBool"] == 1].value_counts(normalize = True)[1]*100)

print ("Percentage of CabinBool = 0 who survived:", train ["Survived"][train["CabinBool"] == 0].value_counts(normalize = True)[1]*100)

#draw a bar plot of CabinBool vs survival
sns.barplot(x="CabinBool", y="Survived", data=train)
plt.show()
#cleaning data
#Let's see how our test data looks 
test.describe(include="all")
#Cain Feature
# We'll start off by dropping the Cabin feature since not a lot more usefull information can be extracted from it. 
train = train.drop(['Cabin'], axis = 1)
test = test.drop(['Cabin'], axis = 1)
#Ticket Feature
# We can also drop the Ticket feature since it's unlikely to yiled any usefull information 
train = train.drop(['Ticket'], axis =1)
test = test.drop(['Ticket'], axis =1)
#Embarked Fature
#now we need to fill in the missing values in the Embarked feature. 
print ("Number of people embarking in Southampton (S):")
southampton = train[train["Embarked"] == "S"].shape[0]
print(southampton)

print("Number of people embarking in Cherbourg (C):")
cherbourg = train[train["Embarked"] == "C"].shape[0]
print(cherbourg)

print("Number of people embarking in Queenstown (Q):")
queenstown = train[train["Embarked"] == "Q"].shape[0]
print(queenstown)
#replacing the missing values in the Embarked feature with s
train = train.fillna({"Embarked": "S"})
#Age Feature
# Next we'll fill in the missing values in the Age feature. Since a higher percentage 
# of values are missing, it would be illogical to fill all of them with the same value 
# (as we did with Embarked). Instead, let's try to find a way to predict the missing ages. 
# create a combined group of both datasets
combine = [train, test]
# extract a title for each Name in the train and test datasets
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract('([A-Za-z]+)\.', expand=False)
    

pd.crosstab(train['Title'], train['Sex'])
#replace various titles with more common names
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Capt', 'Col', 'Don', 'Dr','Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')
    
    dataset['Title'] = dataset['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
#map each of the title groups to a numerical value 
title_mapping = {"Mr": 1, "Miss": 2, "Mrs":3, "Master": 4, "Royal": 5, "Rare": 6}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)
    
train.head()
#fill missing age with node age group for each title
mr_age = train[train["Title"] == 1]["AgeGroup"].mode() #Young Adult
miss_age = train[train["Title"] == 2]["AgeGroup"].mode() #Student
mrs_age = train[train["Title"] == 3]["AgeGroup"].mode() #Adult
master_age = train[train["Title"] == 4]["AgeGroup"].mode() #Baby
royal_age = train[train["Title"] == 5]["AgeGroup"].mode() #Adult
rare_age = train[train["Title"] == 6]["AgeGroup"].mode()  #Adult

age_title_mapping = {1: "Young Adult", 2: "Student", 3: "Adult", 4: "Baby", 5:"Adult", 6: "Adult"}

# I tried to et this code to work with using .map(), but couldn't 
# I've put down a less elegant, temporary solution for now. 
#train = train.fillna({"Age": train["Title"].map(age_title_mapping)})
#test =test.fillna({"Age": test["Title"].map(age_title_mapping)})

for x in range(len(train["AgeGroup"])):
    if train["AgeGroup"][x] == "Unknown":
        train["AgeGroup"][x] = age_title_mapping[train["Title"][x]]
    
for x in range(len(test["AgeGroup"])):
    if test["AgeGroup"][x] == "Unknown":
        test["AgeGroup"][x] = age_title_mapping[test["Title"][x]]
#map each Age value to a numerical value 
age_mapping = {'Baby': 1, 'Child':2, 'Teenager': 3, 'Student': 4, 'Young Adult': 5, 'Adult':6, 'Senior': 7}
train['AgeGroup'] = train['AgeGroup'].map(age_mapping)
test['AgeGroup'] = test['AgeGroup'].map(age_mapping)

train.head()

#dropping the age feature for now, might chage

train = train.drop(['Age'], axis =1)
test= test.drop(['Age'], axis =1)
#Name Feature
# we can drop the name feature now that we've extracted the titles. 
train = train.drop(['Name'], axis =1)
test = test.drop(['Name'], axis = 1)
#Sex Feature 
#Map each Sex value to a numerical value 
sex_mapping = {"male":0, "female":1}
train['Sex'] = train['Sex'].map(sex_mapping)
test['Sex'] = test['Sex'].map(sex_mapping)

train.head()
