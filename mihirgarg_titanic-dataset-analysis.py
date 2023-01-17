##Importing Libraries

import pandas as pd

from pandas import Series, DataFrame
#Importing Dataset

titanic=pd.read_csv('../input/train.csv')

titanic.head()
#Info of Dataset

titanic.info()
#Statistics of Dataset

titanic.describe()
#Importing Libraries for Visualization

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
#Count of male and female in dataset

sns.factorplot('Sex',data=titanic,kind='count',color='red')
#count of male and female by classes

sns.factorplot('Sex',data=titanic,hue='Pclass',kind='count')
#count of male and female by classes

sns.factorplot('Pclass',data=titanic,hue='Sex',kind='count')
def split(passenger):

    age,sex=passenger

    if age < 16:

        return 'child'

    else:

        return sex
titanic['person']=titanic[['Age','Sex']].apply(split,axis=1)

titanic[0:10]
#count of male, female and child by classes

sns.factorplot('Pclass',hue='person',data=titanic,kind='count')
#Distribution of ages (one more method)

titanic['Age'].hist(bins=70,color='blue')
plt.hist(titanic['Age'].dropna(),bins=70)
#mean of ages

titanic['Age'].mean()
#count of male, female and child

titanic['person'].value_counts()
#KDE Plot of age with respect to gender

fig=sns.FacetGrid(titanic,hue='Sex',aspect=4)

fig.map(sns.kdeplot,'Age',shade=True)

oldest = titanic['Age'].max()

fig.set(xlim=(0,oldest))

fig.add_legend()
#KDE Plot of age with respect to person

fig=sns.FacetGrid(titanic,hue='person',aspect=4)

fig.map(sns.kdeplot,'Age',shade=True)

oldest = titanic['Age'].max()

fig.set(xlim=(0,oldest))

fig.add_legend()
#KDE Plot of age with respect to Pclass

fig=sns.FacetGrid(titanic,hue='Pclass',aspect=4)

fig.map(sns.kdeplot,'Age',shade=True)

oldest = titanic['Age'].max()

fig.set(xlim=(0,oldest))

fig.add_legend()
titanic.head()
deck=titanic['Cabin'].dropna()
deck.head()
levels = []



for level in deck:

    levels.append(level[0])    



cabin = DataFrame(levels)

cabin.columns = ['Cabin']

sns.factorplot('Cabin',data=cabin,palette='winter_d',kind='count')
cabin=cabin[cabin.Cabin!='T']
sns.factorplot('Cabin',data=cabin,palette='rainbow',kind='count')
titanic.head()
#plot of passengers embarked from different places

sns.factorplot('Embarked',data=titanic,kind='count',color='blue')
#count of embarkment by classes

sns.factorplot('Embarked',hue='Pclass',data=titanic,kind='count',palette='ocean')
titanic.head()
titanic['Alone']=titanic.SibSp+titanic.Parch
titanic.head()
titanic['Alone']
titanic['Alone'].loc[titanic['Alone']>0]='With Family'

titanic['Alone'].loc[titanic['Alone']==0]='Alone'
titanic.head()
# count of passengers with family or alone

sns.factorplot('Alone',data=titanic,kind='count',palette='hot')
titanic['Survivor']=titanic.Survived.map({0:'No',1:'Yes'})

titanic.head()
#count of passengers who survived or not

sns.factorplot('Survivor',data=titanic,kind='count')
#count of passengers who survived or not by classes

sns.factorplot('Pclass','Survived',data=titanic)
#count of passengers who survived or not by person

sns.factorplot('Pclass','Survived',hue='person',data=titanic)

#count of passengers who survived or not with respect to age

sns.lmplot('Age','Survived',data=titanic)
#count of passengers who survived or not with respect to age by classes

sns.lmplot('Age','Survived',hue='Pclass',data=titanic,palette='winter')
generations=[10,20,40,60,80]

sns.lmplot('Age','Survived',hue='Pclass',data=titanic,palette='summer',x_bins=generations)
#count of passengers who survived or not with respect to age by gender

sns.lmplot('Age','Survived',hue='Sex',data=titanic,palette='winter',x_bins=generations)
def decklevel(cabin):

    deck=cabin

    for level in deck:

        levels=level[0]

        return levels

titanic['decklevels'] = titanic[['Cabin']].dropna().apply(decklevel,axis=1)
titanic.head()
# plot of survivors by decklevels

sns.factorplot('decklevels','Survived',data=titanic)
titanic=titanic[titanic.Cabin!='T']
sns.factorplot('decklevels','Survived',data=titanic)
# plot of survivors by decklevels with respect to person

sns.factorplot('decklevels','Survived',hue='person',data=titanic)
# plot of survivors by decklevels with respect to pclass

sns.factorplot('decklevels','Survived',hue='Pclass',data=titanic)
# plot of survivors by decklevels with respect to sex

sns.factorplot('decklevels','Survived',hue='Sex',data=titanic)
#plot of surviors with family or not

sns.factorplot('Alone','Survived',data=titanic)
#plot of surviors with family or not by sex

sns.factorplot('Alone','Survived',hue='Sex',data=titanic)
#plot of surviors with family or not by pclass

sns.factorplot('Alone','Survived',hue='Pclass',data=titanic)
#plot of surviors with family or not by decklevels

sns.factorplot('Alone','Survived',hue='decklevels',data=titanic)