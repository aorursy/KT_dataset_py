import pandas as pd

import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
train = pd.read_csv('../input/introductiontopandas/titanic_train.csv')
train.head()
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
#Purple is good and yello is null
sns.set_style('whitegrid')
#Who survived vs. didn't...use count plot
sns.countplot(x='Survived',data=train)
#Remember 1 signifies they survived
#Adding Gender
sns.countplot(x='Survived',hue='Sex',data=train,palette='RdBu_r')
sns.countplot(x='Survived',hue='Pclass',data=train)
sns.distplot(train['Age'].dropna(),kde=False,bins=30)
#Easy way to drop nulls above
train['Age'].plot.hist(bins=35)
train.info()
sns.countplot(x='SibSp',data=train)
#Number of siblings or spouses on board - No spouses or children most common
#Next highest is couples
train['Fare'].hist(bins=40,figsize=(10,4))
#How much paying...cheaper fares
#
#
#Part 2 - Logistic Regression
#Above lots of nulls in the purple/yellow diagram focused on nulls.
plt.figure(figsize=(10,7))

sns.boxplot(x='Pclass',y='Age',data=train)
#Class of individuals.....looking at Age of each classes
#So first class and second class passengers were older
#Taking care of nulls putting in averages for blanks else using age
def impute_age(cols):

    Age = cols[0]

    Pclass = cols[1]

    

    if pd.isnull(Age):

        

        if Pclass == 1:

            return 37

        elif Pclass == 2:

            return 29

        else:

            return 24

    else:

        return Age
train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)
sns.heatmap(train.isnull(),yticklabels=False,cbar=False)
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
train.columns
train.dropna(inplace=True)
#Taking out more NAs and missing values
#Machine learning can't take male or female so making new column 1/0
#Dummies looks at column and gets dummy values
pd.get_dummies(train['Sex'])
#Problem with this.....Multi Collinearity....Predicting if not female male
#One column predicts other. Feed both then predicts other.
#This will drop the first column brining back only one.
sex = pd.get_dummies(train['Sex'],drop_first=True)
embark = pd.get_dummies(train['Embarked'],drop_first=True)
embark.head()
train = pd.concat([train,sex,embark],axis=1)
#Axis = 1 adds as new column
train.head(2)
#New columns are indicator or dummy columns meaning we don't need others
train.drop('PassengerId',axis=1,inplace=True)
train.head()