# Import necessary libraries



# Import pandas to handle the data

import pandas as pd



# Import numpy and matplotlib for analysis

import numpy as np

import matplotlib.pyplot as plt



# Import seaborn because it's pretty and I like it

import seaborn as sns

sns.set_style('white')



# Import things that get stuff done (aka Machine Learning libraries)

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

# Import the Titanic Data Set

titanic= pd.read_csv("../input/train.csv")



# Import the data set we will use for testing

test= pd.read_csv("../input/test.csv")



# Let's check to see what it looks like...

titanic.info()
titanic= titanic.drop(['PassengerId','Ticket'], axis=1)

test=test.drop(['PassengerId','Ticket'], axis=1)



titanic.head()
print (titanic[["Sex","Survived"]].groupby(['Sex'], as_index=False).mean())
with_family=titanic[['Survived','SibSp']].groupby(['Survived'],as_index=False)

sns.barplot(x='Survived', y='SibSp', data=with_family, order=[1,0])

family=titanic['SibSep']
