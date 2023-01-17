import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import csv as csv



train_df = pd.read_csv('../input/train.csv',index_col='PassengerId')
#Create a new function:

def num_missing(x):

  return sum(x.isnull())



#Missing values per column

#axis=0 defines that function is to be applied on each column

train_df.apply(num_missing, axis=0) 





                        


#sex - mapping 0 and 1

train_df['Gender'] = train_df['Sex'].map({'female':0, 'male':1})



#Embarked - filling empty with highest occurance  -- 'S'

#train_df.groupby("Embarked").size()

if len(train_df['Embarked'][train_df['Embarked'].isnull()]) > 0:

    train_df['Embarked'][train_df['Embarked'].isnull()] = train_df['Embarked'].dropna().mode().values    



#Title

train_df['Title'] = train_df['Name'].str.split('.').str.get(0).str.split(', ').str.get(1)

train_df['Title'].str.replace('Ms','Miss')

train_df['Title'].str.replace('Lady','Mrs')



#Age

#t = train_df.groupby(['Title'])

#t['Age'].mean()

#if len(train_df['Age'][train_df['Age'].isnull()]) > 0:

p = train_df.loc[train_df['Age'].isnull()]['Title'].index

#train_df.loc[p]['Title']

#train_df['Title'][train_df['Age'].isnull()]

    



#show cleaned data

#train_df
