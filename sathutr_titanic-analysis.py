import numpy as np 

import pandas as pd 

import missingno as msno

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

import re

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix 

from sklearn import metrics
# Importing data in dataframes using labelled test set for a change

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

survival = pd.read_csv("../input/gender_submission.csv")
train.head()
train.info()
msno.matrix(train)
# Percentage of survivors and non-survivors

train["Survived"].value_counts()
#Pie representing percentage of survival and non-survival on Titanic 

labels = 'Survived','Did Not Survive'

sizes = [549,342]

colours = ['mediumseagreen','slategray']

explode = (0.1, 0)



# Plot

plt.pie(sizes, explode=explode, labels=labels, colors=colours,

autopct='%1.1f%%', shadow=True, startangle=140)

plt.axis('equal')

plt.tight_layout()

plt.show()
# How likely is it to survive if one is a male or female of a particular age?



fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(10, 4))



men = train[train['Sex'] == 'male']

woman = train[train['Sex'] == 'female']



ax = sns.distplot(woman[woman['Survived']==1].Age.dropna(), bins=18, label = 'survived', ax = axes[0], kde =False)

ax = sns.distplot(woman[woman['Survived']==0].Age.dropna(), bins=40, label = 'not_survived', ax = axes[0], kde =False)

ax.legend()

ax.set_title('Female')



ax = sns.distplot(men[men['Survived']==1].Age.dropna(), bins=18, label = 'survived', ax = axes[1], kde =False)

ax = sns.distplot(men[men['Survived']==0].Age.dropna(), bins=40, label = 'not_survived', ax = axes[1], kde =False)

ax.legend()

ax.set_title('Male')
#PClass 

sns.barplot(x='Pclass', y='Survived', data=train)
# Class, survival rate and sex 



g = sns.catplot(x="Sex", y="Survived", col="Pclass", data=train, saturation=.5, kind="bar", ci=None, aspect=.6)



(g.set_axis_labels("", "Survival Rate")

.set_xticklabels(["Men", "Women", "Children"])

.set_titles("{col_name} {col_var}")

.set(ylim=(0, 1))

.despine(left=True))
# Class Vs Survival

grid = sns.FacetGrid(train, col='Survived', row='Pclass', size=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend();
# From this point, I'll be working on training and test set together so as to maintain consistency with preprocessing

test_set = pd.merge(test,survival,how='left',on='PassengerId')

test_set.head()



#Combining train and test set for preprocessing 

data = pd.concat([train, test_set], sort=False, names=['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp','Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked', 'Survived'])
# Dealing with the 'Cabin' column 

# Getting the Dock code from the cabin number 



data['Decktemp'] = data['Cabin'].str.replace('[^a-zA-Z]', '')

data['Decktemp'].value_counts()
# Creating new column for docks

data['Deck'] = data['Decktemp'].astype(str).str[0]



#Giving numerical codes to Docks 

deck = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "T": 8}

data['Deck'] = data['Deck'].map(deck)

data['Deck'] = data['Deck'].fillna(0)

data['Deck'] = data['Deck'].astype(int)
# Now drop the cabin feature

data = data.drop(['Cabin', 'Decktemp'], axis=1)

data.columns
data['Age'].describe()
#Dealing with missing values in 'Age'

#Substituting MV with mean

data['Age'] = data['Age'].fillna(29.8)
# Dealing with MV in Embarked column

# Substituting MV with most occured value 

data['Embarked'] = data['Embarked'].fillna('S')



# Converting 'Embarked' column to numerical values 

embark = {"S": 1, "C": 2, "Q": 3}

data['Embarked'] = data['Embarked'].map(embark)
# Sex - numbering the classes in the feature

sex = {"male": 0, "female": 1}

data['Sex'] = data['Sex'].map(sex)

data.head()
# Creating relatives column using 'Sibsp' 'Parch'

data['Relatives'] = data['SibSp']+data['Parch']
# Fare 

# Filling missing entry in 'Fare'

data[data.isnull().any(axis=1)]
# Based on the Pclass - Filling the fare to first quartile value - 7.8

data['Fare'] = data['Fare'].fillna(7.8)



#Binning values in Fare 0 to 515

bins = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 400, 420, 440, 460, 480, 500, 520]

labels = list(range(1,27))

data['Fare_cat'] = pd.cut(data['Fare'], bins=bins, labels=labels, include_lowest=True)



#Drop 'Fare' column 

data = data.drop(['Fare'], axis=1)
# Categorizing 'Age' column



bins = [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80]

labels = list(range(1,17))

data['Age_cat'] = pd.cut(data['Age'], bins=bins, labels=labels, include_lowest=True)



#Drop 'Age' column 

data = data.drop(['Age'], axis=1)
data = data.drop(['Name','Ticket'], axis=1)
# Splitting Train and Test Set

train_passengers = list(train['PassengerId'])

train = data[data['PassengerId'].isin(train_passengers)]

test_passengers = list(test['PassengerId'])

test = data[data['PassengerId'].isin(test_passengers)]
train = train.drop(['PassengerId'], axis=1)

test = test.drop(['PassengerId'], axis=1)
#Splitting train and test set 

X_train = train.drop("Survived", axis=1)

Y_train = train["Survived"]

X_test  = test.drop("Survived", axis=1)

Y_test = test["Survived"]
#Random forest 

random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)



Y_prediction = random_forest.predict(X_test)



random_forest.score(X_train, Y_train)

acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
#Accuracy

acc_random_forest