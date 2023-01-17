# Import dependencies

import numpy as np

import pandas as pd 
# Data import

titanic_train = pd.read_csv("../input/train.csv")
titanic_train.head()
titanic_train.columns
titanic_train.shape
survived_class = pd.crosstab(index=titanic_train["Survived"],columns=titanic_train["Pclass"], margins=True)   # Including row and column totals
survived_class.head()
# Renaming the columns to

survived_class.columns = ["class1","class2","class3","rowtotal"]

survived_class
# Renaming the index

survived_class.index= ["died","survived","columntotal"]

survived_class
surv_sex_class = pd.crosstab(index=titanic_train["Survived"], columns=[titanic_train["Pclass"],titanic_train["Sex"]],margins=True)

surv_sex_class.head()
surv_sex_class.index= ["died","survived","columntotal"]

surv_sex_class
#Plot the count distribution (Bernoulli) of survival  (sns.countplot) 

import seaborn as sns

import matplotlib.pyplot as plt



plt.figure(figsize=(6,10))

sns.set(style="darkgrid")

m = sns.countplot(x="Survived", data=titanic_train)

plt.show()
plt.figure(figsize=(15,8))

m = sns.countplot(x="Pclass", hue="Survived", data=titanic_train)
plt.figure(figsize=(15,8))

graph = sns.countplot(x="Survived",hue="Sex" ,data=titanic_train)

plt.show()
plt.figure(figsize=(12,6))

sns.catplot(x='Sex', y='Survived', hue='Pclass', data=titanic_train, kind='bar')

plt.ylabel("Survival Rate")

plt.show()
plt.figure(figsize=(12,6))

sns.catplot(x="Sex",hue='Pclass' ,kind="count", data=titanic_train)

plt.show()
survived_sib_class = pd.crosstab(titanic_train["SibSp"], titanic_train["Parch"], margins=True) 
plt.figure(figsize=(12,6))

sns.boxplot('SibSp','Survived',data=titanic_train)

plt.title('Distribution of Sibsip')

plt.show()
# SibSp vs Survived per gender

plt.figure(figsize=(12,6))

sns.catplot(x="SibSp", y="Survived", hue="Sex",col="Pclass", data=titanic_train, kind='bar')

plt.show()
# Pclass  wise

f,ax=plt.subplots(1,3,figsize=(20,5))

titanic_train['Pclass'].value_counts().plot.bar(ax=ax[0])

ax[0].set_title('Passengers with Pclass')

ax[0].set_ylabel('Count')

sns.countplot('Pclass',hue='Survived',data=titanic_train,ax=ax[1])

ax[1].set_title('Survived with Pclass')

sns.barplot(x="Pclass", y="Survived", data=titanic_train,ax=ax[2])

ax[2].set_title('Survived in Pclass')

plt.show()
# Gender wise 

f,ax=plt.subplots(1,3,figsize=(20,5))

titanic_train[['Sex','Survived']].groupby(['Sex']).mean().plot.bar(ax=ax[0])

ax[0].set_title('Servived : gender')

sns.countplot('Sex',hue='Survived',data=titanic_train,ax=ax[1])

ax[1].set_title('Survived : gender')

sns.barplot(x="Sex", y="Survived", data=titanic_train,ax=ax[2])

ax[2].set_title('Survived : Gender')

plt.show()
plt.subplots(figsize=(15,20))

sns.boxplot('Pclass','Fare',data=titanic_train, hue= "Survived")

plt.title('Fares distribution')

plt.show()