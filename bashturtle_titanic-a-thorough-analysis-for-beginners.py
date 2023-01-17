#import libraries for data preprocessing

import pandas as pd

import numpy as np

import seaborn as sns

import random as rnd

import matplotlib.pyplot as plt

#import machine learning models ,no need to understand the different classifiers now 

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

#load data 

train_df=pd.read_csv('../input/train.csv')

test_df=pd.read_csv('../input/test.csv')

#create a pipeline for sumltaneous processing 

combine=[train_df,test_df]

print(combine[0].shape)

print(combine[1].shape)
#showing columns in the dataset 

print(train_df.columns.values)
train_df
train_df.info()

test_df.info()
train_df.describe()
train_df.head()
train_df[['Pclass','Survived']].groupby(['Pclass'],as_index=False).mean().sort_values(by='Survived',ascending=False)
train_df[['SibSp','Survived']].groupby(['SibSp'],as_index=False).mean().sort_values(by='Survived',ascending=False)
train_df[['Parch','Survived']].groupby(['Parch'],as_index=False).mean().sort_values(by='Survived',ascending=False)
train_df[['Embarked','Survived']].groupby(['Embarked'],as_index=False).mean().sort_values(by='Survived',ascending=False)
new_features=train_df

new_features['FamilySize']=train_df['Parch']+train_df['SibSp']+1

new_features.FamilySize.nunique()

new_features[['FamilySize','Survived']].groupby(['FamilySize'],as_index=False).mean().sort_values(by='Survived',ascending=False)
train_df[['Sex','Survived']].groupby(['Sex'],as_index=False).mean().sort_values(by='Survived',ascending=False)
g=sns.FacetGrid(train_df,col='Survived')

g.map(plt.hist, 'Age', bins=20)
new_features=train_df

new_features.Fare=new_features.Fare.astype(int)

g=sns.FacetGrid(new_features,col='Survived')

g.map(plt.hist, 'Fare', bins=20)
#Plotting age with survival

#grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)

grid=sns.FacetGrid(train_df,col='Survived')

grid.map(sns.distplot,'Age',kde=False,bins=20)
train_df.head()
#plotting pclass with survival variations

sns.catplot(x='Pclass',y='Survived',kind='bar',data=train_df,)

sns.catplot(x='Pclass',y='Fare',kind='bar',data=train_df,)

sns.catplot(x='Pclass',y='Survived',kind='bar',hue='Sex',data=train_df)
sns.catplot(x='Pclass',hue='Sex',kind='count',data=train_df)
grid=sns.FacetGrid(train_df,col='Pclass',row='Sex')

grid.map(sns.distplot,'Age',kde=False,bins=20)