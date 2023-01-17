# We can use the pandas library in python to read the csv file.

import pandas as pd

# for numerical computations we can use numpy library in python

import numpy as np

# libraries for Data Visualization

import matplotlib.pyplot as plt

import seaborn as sns
# Loading train, test & gender_submission data...
# Here, creating a pandas dataframe and assigning to the variables: titanic_train, titanic_test & titanic_gen

titanic_train = pd.read_csv("../input/titanic/train.csv")
titanic_test = pd.read_csv("../input/titanic/test.csv")
titanic_gen = pd.read_csv("../input/titanic/gender_submission.csv")

# here the dataframe output of train file:

titanic_train
# here the dataframe output of test file:

titanic_test
# The 'shape' command will give number of rows and columns in dataset, train & test dataset are multidimentional data.
# (rows, columns)
# To see the shape of train & test dataset:

print('shape of train:',titanic_train.shape)
print('shape of test:',titanic_test.shape)
# we can see that brief information about 'train & test' dataset using command 'info()'
# total values in each column, null/not null, memory occupied, datatype etc.

titanic_train.info()
# similarly:

titanic_test.info()
# command 'describe()' funcion computes a summary of statistics pertaining to the DataFrame columns. This function gives the mean, std and IQR values. And, function excludes the character columns and given summary about numeric columns.

titanic_train.describe()
# similarly:

titanic_test.describe()
# let's see if there is any more columns with missing values, need command 'isnull()':
# train: Age, Cabin and Embarked have missing values.

titanic_train.isnull().sum()
#similarly:
# test: Age, Fare and Cabin have missing values.

titanic_test.isnull().sum()
# let's print first five lines of our train dataset.

titanic_train.head()
# let's print first five lines of our test dataset.

titanic_test.head()
# Data Visualization is very important in Data Science Project.
# By Data Visualization we can understand the data quickly and efficiently.
# A Diagram can convey more meaning when compared to 1000 written words.
# bar plot visualization for Sex vs Survived:
# from the below visualisation, it shows that women are more likely survied then men.

sns.barplot(x = 'Sex', y = 'Survived', data = titanic_train)
# survival percentage of female:
print('Percentage of survived females:', titanic_train['Survived'][titanic_train['Sex']=='female'].value_counts(normalize = True)[1]*100)

# survival percentage of male:
print('Percentage of survived men:', titanic_train['Survived'][titanic_train['Sex']=='male'].value_counts(normalize = True)[1]*100)
# bar plot visualization for SibSp vs Survived:
# SibSp: Siblings/Spouses
# The following plot shows a person aboarded with more than 2 Siblings/Spouses more likely survived whereas a person aboarded without siblings or spouse more likely dead

sns.barplot(x = 'SibSp', y = 'Survived', data = titanic_train)
# survival percentage of Siblings/Spouses Aboard

print('Survived percentage of Siblings/Spouses Aboard-0:', titanic_train['Survived'][titanic_train['SibSp']==0].value_counts(normalize=True)[1]*100)

print('Survived percentage of Siblings/Spouses Aboard-1:', titanic_train['Survived'][titanic_train['SibSp']==1].value_counts(normalize=True)[1]*100)

print('Survived percentage of Siblings/Spouses Aboard-2:', titanic_train['Survived'][titanic_train['SibSp']==2].value_counts(normalize=True)[1]*100)

print('Survived percentage of Siblings/Spouses Aboard-3:', titanic_train['Survived'][titanic_train['SibSp']==3].value_counts(normalize=True)[1]*100)

print('Survived percentage of Siblings/Spouses Aboard-4:', titanic_train['Survived'][titanic_train['SibSp']==4].value_counts(normalize=True)[1]*100)
# bar plot visualization for Pclass vs Survived
# Following plot confirms 1st class more likely survived than other classes whereas 3rd class more likely dead than other classes.

sns.barplot(x='Pclass', y='Survived', data=titanic_train)
# survival percentage of Pclass:

print('Percentage of Pclass-1 who survived:', titanic_train['Survived'][titanic_train['Pclass']==1].value_counts(normalize=True)[1]*100)

print('Percentage of Pclass-2 who survived:', titanic_train['Survived'][titanic_train['Pclass']==2].value_counts(normalize=True)[1]*100)

print('Percentage of Pclass-3 who survived:', titanic_train['Survived'][titanic_train['Pclass']==3].value_counts(normalize=True)[1]*100)
# bar plot visualization for Parch vs Survived
# Parch: Parents/Children Aboard
# Following plot confirms a person aboarded with more than 2 Parents/Children are more likely survived whereas a person aboarded alone more likely dead.

sns.barplot(x='Parch', y='Survived', data=titanic_train)
# survival percentage of Parch(Parents/Children Aboard):

print('Survived percentage of Parents/Children-0:', titanic_train['Survived'][titanic_train['Parch']==0].value_counts(normalize=True)[1]*100)

print('Survived percentage of Parents/Children-1:', titanic_train['Survived'][titanic_train['Parch']==1].value_counts(normalize=True)[1]*100)

print('Survived percentage of Parents/Children-2:', titanic_train['Survived'][titanic_train['Parch']==2].value_counts(normalize=True)[1]*100)

print('Survived percentage of Parents/Children-3:', titanic_train['Survived'][titanic_train['Parch']==3].value_counts(normalize=True)[1]*100)

print('Survived percentage of Parents/Children-5:', titanic_train['Survived'][titanic_train['Parch']==5].value_counts(normalize=True)[1]*100)
# bar plot visualization for Embarked vs Survived
# Following plot shows people aboarded from C better survived than people aboarded from Q likely dead and people aboarded from S more likely dead.

sns.barplot(x='Embarked', y='Survived', data=titanic_train)