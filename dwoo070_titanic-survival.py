# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



            

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')



combine = [train_df,test_df]
train_df.head()
train_df.info()
sns.FacetGrid(train_df, col = 'Pclass').map(plt.hist,'Age',bins = 20)
g = sns.FacetGrid(train_df, col = 'Survived', row = 'Pclass')

g.map(plt.hist, 'Age', bins = 20)
train_df = train_df.drop(['Ticket','Cabin'], axis = 1)

test_df = test_df.drop(['Ticket','Cabin'], axis = 1)

combine = [train_df,test_df]
for dataset in combine:

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)



for dataset in combine:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\

 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')



    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    

train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
train_df =train_df.drop(['Name','PassengerId'],axis = 1)

test_df = test_df.drop(['Name'],axis = 1)

combine = [train_df,test_df]
for dataset in combine:

    dataset['Sex'] = dataset['Sex'].map({'female':1, 'male':0}).astype(int)
train_df.head()
mask = (train_df['Pclass'])==1 & (train_df['Sex'] ==1)
for dataset in combine:

    for i in range(0,2):

        for j in range(0,3):

            mask = (train_df['Sex'])== i & (train_df['Pclass'] == j + 1)

            guess = int(train_df['Age'][mask].median()/0.5 + 0.5 ) * 0.5

            dataset.loc[(dataset['Age'].isnull()) & mask, 'Age'] = guess

            

            

            

        

       

train_df['AgeBand']=pd.cut(train_df['Age'],5)
(train_df.Survived.groupby(train_df.AgeBand)).mean()
for dataset in combine:    

    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

    dataset.loc[ dataset['Age'] > 64, 'Age']

train_df.head()
train_df = train_df.drop('AgeBand',axis = 1)
train_df.head()
combine = [train_df,test_df]

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

for dataset in combine:

    dataset['Title'] = dataset['Title'].map(title_mapping)

    dataset['Title'] = dataset['Title'].fillna(0)



train_df.head()
for dataset in combine:

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1





train_df['Survived'].groupby(train_df.FamilySize).mean().sort_values( ascending = False)
for dataset in combine:

    dataset['IsAlone'] = 0

    dataset.loc[dataset['FamilySize']==1, 'IsAlone'] = 1

    

    
train_df = train_df.drop(['Parch','FamilySize','SibSp'],axis = 1)

test_df = test_df.drop(['Parch','FamilySize','SibSp'],axis = 1)
train_df['Pclass'] = train_df['Pclass'].astype(int)

test_df['Pclass'] = test_df['Pclass'].astype(int)
for dataset in combine:

    dataset['AgeClass'] = dataset.Age*dataset.Pclass



#train_df.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10)