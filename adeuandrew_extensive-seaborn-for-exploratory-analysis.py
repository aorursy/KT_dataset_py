import pandas as pd 

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

#1. Loading train and test data sets

train=pd.read_csv('../input/train.csv')

test=pd.read_csv('../input/test.csv')

print (train.shape)

print (test.shape)

train.columns
sns.countplot(x='Sex',data=train)

print ('Total number of passengers ',train.shape[0])
sns.barplot(x='Sex',y='Survived',data=train)

train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
sns.barplot(x='Pclass',y='Survived',data=train)

sns.factorplot('Pclass','Survived',data=train)
train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
sns.barplot(x='Pclass',y='Survived', hue='Sex',data=train)
sns.barplot(x="Sex", y="Survived", hue="Pclass", data=train);
sns.countplot(x='Embarked', data=train)
sns.countplot(x='Embarked',hue='Pclass',data=train)
sns.barplot(x='Embarked',y='Survived',hue='Pclass',data=train)
sns.countplot(x='Survived',hue='Embarked',data=train)
sns.countplot(x='SibSp',data=train)
sns.barplot(x='SibSp',y='Survived',data=train)
sns.barplot(x='SibSp',y='Survived',hue='Pclass',data=train)
sns.countplot(x='SibSp',hue='Pclass',data=train)
train['SibSp'].max() # max equal 8

bins = [-1,0,3,10]  

labels = ['alone', 'small family', 'big family']

sb_groups = pd.cut(train.SibSp, bins, labels=labels,right=True)

train['FamilyType'] = sb_groups
sns.barplot(x='FamilyType',y='Survived',data=train)
sns.factorplot('FamilyType','Survived',data=train)
sns.countplot(x='FamilyType',hue='Pclass',data=train)
sns. barplot(x='FamilyType',y='Survived',hue='Sex',data=train)
sns.barplot(x='Parch', y ='Survived',hue='FamilyType', data=train)
sns.distplot(train['Age'].dropna(),bins=30)


facet = sns.FacetGrid( train , hue='Survived' , aspect=4 , row = None , col = None )

facet.map( sns.kdeplot ,'Age' , shade= True )

facet.set( xlim=( 0 , train['Age'].max() ) )

facet.add_legend()
grid = sns.FacetGrid(train, row='Survived', col='Sex', size=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=.5, bins=30)

grid.add_legend();
g = sns.FacetGrid(train, row='Survived', col='Pclass')

g.map(sns.distplot, "Age")
fig = sns.FacetGrid(train,hue='Pclass',aspect=4)

fig.map(sns.kdeplot,'Age',shade='True')

oldest = train['Age'].max()

fig.set(xlim=(0,oldest))

fig.add_legend()


bins = [0, 12, 17, 60,90]

labels = ['child', 'teenager', 'adult', 'elder']

age_groups = pd.cut(train.Age.dropna(), bins, labels=labels)

train['Age_group'] = age_groups

sns.barplot(x='Age_group',y='Survived',hue='Sex',data=train)
corrmat = train[["Age","SibSp","Fare","Parch"]].corr()

sns.heatmap(corrmat, vmax=1, square=True)
sns.jointplot(data=train, x='Pclass', y='Fare', kind='reg', color='g')

sns.plt.show()
sns.boxplot(x="Pclass", y="Fare", hue='Sex', data=train)
df = train.pivot_table(index='Embarked', columns='Age_group', values='Fare', aggfunc=np.median)

sns.heatmap(df, annot=True, fmt=".1f")

sns.barplot(x='Cabin',y='Survived',data=train)
train.Cabin.unique()
train.Cabin.isnull().sum()
train['CabinLetter'] = train['Cabin'].apply(lambda x:x[0] if type(x)==str else 'NaN' )

sns.barplot(x='CabinLetter',y='Survived',data=train)
sns.barplot(x='CabinLetter',y='Survived',hue='Pclass',data=train)
sns.barplot(x='CabinLetter',y='Survived',hue='FamilyType',data=train)