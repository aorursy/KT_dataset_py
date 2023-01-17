# Imports



# pandas

import pandas as pd

from pandas import Series,DataFrame



# numpy, matplotlib, seaborn

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

%matplotlib inline



# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB



# get titanic & test csv files as a DataFrame

titanic_df = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )

test_df    = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )



# preview the data

titanic_df.head()

titanic_df = titanic_df.drop(['PassengerId','Name','Ticket'], axis=1)

#titanic_df["Embarked"] = titanic_df["Embarked"].fillna("S")
 

# plot

sns.factorplot('Embarked','Survived', data=titanic_df,size=3,aspect=3)

sns.factorplot('Pclass','Survived', data=titanic_df,size=3,aspect=3)

sns.factorplot('Sex','Survived', data=titanic_df,size=3,aspect=3)



#plt.plot("Age", "Survived")



fig, (axis1,axis2,axis3) = plt.subplots(1,3,figsize=(12,4))

sns.countplot(x='Survived', hue="Pclass", data=titanic_df, order=[1,0], ax=axis1)

sns.countplot(x='Survived', hue="Embarked", data=titanic_df, order=[1,0], ax=axis2)

sns.countplot(x='Survived', hue="Sex", data=titanic_df, order=[1,0], ax=axis3)



fig, (axis1,axis2,axis3) = plt.subplots(1,3,figsize=(12,4))

sns.countplot(x='Survived', hue="SibSp", data=titanic_df, order=[1,0], ax=axis1)

sns.countplot(x='Survived', hue="Parch", data=titanic_df, order=[1,0], ax=axis2)

sns.countplot(x='Pclass', hue="Parch", data=titanic_df, ax=axis3)



fig, (axis1,axis2,axis3) = plt.subplots(1,3,figsize=(12,4))

sns.countplot(x='SibSp', hue="Sex", data=titanic_df, ax=axis1)

sns.countplot(x='Parch', hue="Sex", data=titanic_df, ax=axis2)

sns.countplot(x='Pclass', hue="Parch", data=titanic_df, ax=axis3)
embark_dummies_titanic  = pd.get_dummies(titanic_df['Embarked'])

embark_dummies_titanic.drop(['S'], axis=1, inplace=True)

def print_full(x):

    pd.set_option('display.max_rows', len(x))

    print(x)

    pd.reset_option('display.max_rows')

print_full(embark_dummies_titanic)