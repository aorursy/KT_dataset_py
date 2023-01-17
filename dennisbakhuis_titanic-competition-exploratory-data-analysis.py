import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns


train = pd.read_csv('/kaggle/input/titanic/train.csv')
test = pd.read_csv('/kaggle/input/titanic/test.csv')
trainnRows, trainnCols = train.shape
testnRows, testnCols = test.shape
print('Train rows:', trainnRows)
print('Train columns:', trainnCols)
print('Test rows:', testnRows)
print('Test columns:', testnCols)
train.info()
print('Train:')
print(train.isna().sum())
print('\nTest:')
print(test.isna().sum())
print('Train sample:')
print(train.sample(5))
print('\nTest sample:')
print(test.sample(5))
train.apply(lambda x: len(x.unique()), axis=0)
print('Unique values for Survided:', train['Survived'].unique())
print('Unique values for Pclass:', train['Pclass'].unique())
print('Unique values for Sex:', train['Sex'].unique())
print('Unique values for SibSp:', train['SibSp'].unique())
print('Unique values for Parch:', train['Parch'].unique())
print('Unique values for Embarked:', train['Embarked'].unique())
train['Name'].sample(10)
train['title'] = train.Name.str.split().apply(lambda x: [y for y in x if y[-1] =='.'][0])
train.title.unique()
train.groupby('title').PassengerId.count()
train.loc[(train.Name.str.contains('\('))].head()
train.loc[1].Name
train.loc[train.Name.str.contains('\"')]
train['buyersurname'] = train.Name.apply(lambda x: x.split(',')[0])
train['surname'] = train.buyersurname # by default it is the same person
train['maidenname'] = None

boughtForOthers = train.loc[(train.Name.str.contains('\(')) & ~(train.Name.str.contains('\"'))] # bought for others
# first the none wifes:
names = boughtForOthers.loc[boughtForOthers.title !='Mrs.'].Name.apply(lambda x: x.split('(')[1].split(')')[0].split()[-1])
train.loc[names.index, 'surname'] = names
# now the maiden names of the wifes
names = boughtForOthers.loc[boughtForOthers.title =='Mrs.'].Name.apply(lambda x: x.split('(')[1].split(')')[0])
names = names.loc[names.apply(lambda x: len(x.split())>1)]
train.loc[names.index, 'maidenname'] = names
train.sample(5)
def proc_names(inputdf):
    df = inputdf.copy()
    df['title'] = df.Name.str.split().apply(lambda x: [y for y in x if y[-1] =='.'][0])
    df['buyersurname'] = df.Name.apply(lambda x: x.split(',')[0])
    df['surname'] = df.buyersurname # by default it is the same person
    df['maidenname'] = None

    boughtForOthers = df.loc[(df.Name.str.contains('\(')) & ~(df.Name.str.contains('\"'))] # bought for others
    # first the none wifes:
    names = boughtForOthers.loc[boughtForOthers.title !='Mrs.'].Name.apply(lambda x: x.split('(')[1].split(')')[0].split()[-1])
    df.loc[names.index, 'surname'] = names
    # now the maiden names of the wifes (this is not perfect and there are some errors)
    names = boughtForOthers.loc[boughtForOthers.title =='Mrs.'].Name.apply(lambda x: x.split('(')[1].split(')')[0])
    names = names.loc[names.apply(lambda x: len(x.split())>1)]
    df.loc[names.index, 'maidenname'] = names
    return df
    


ticketSplit = train.Ticket.apply(lambda x: x.split()).values
train['ticketnumber'] = pd.array([int(x[-1]) if x[-1].isdigit() else None for x in ticketSplit], dtype=pd.Int64Dtype())
train['ticketprefix'] = [np.nan if len(x)==1 else ' '.join(x[:len(x)-1]) for x in ticketSplit]
train.ticketprefix.unique()
t = train.groupby('ticketnumber').PassengerId.count().sort_values(ascending=False)
t[t>1]
train.sort_values('ticketnumber')
def proc_tickets(inputdf):
    df = inputdf.copy()
    ticketSplit = df.Ticket.apply(lambda x: x.split()).values
    df['ticketnumber'] = pd.array([int(x[-1]) if x[-1].isdigit() else None for x in ticketSplit], dtype=pd.Int64Dtype())
    df['ticketprefix'] = [np.nan if len(x)==1 else ' '.join(x[:len(x)-1]) for x in ticketSplit]
    return df

train['Age'].unique()
train.loc[train.Age< 1]
train.Age.hist(bins=10)
train.loc[train.Pclass==1].Age.hist(bins=10)
train.loc[train.Pclass==3].Age.hist(bins=10)
train.loc[(train.Pclass==1) & (train.Sex=='male')].Age.hist(bins=10)
train.loc[(train.Pclass==1) & (train.Sex=='female')].Age.hist(bins=10)
train['Age'] = train.groupby(['Sex', 'Pclass'])['Age'].apply(lambda group: group.fillna(group.median()))
def proc_age_single(inputdf):
    """
    Return dataframe with missing ages filled using the median from groups Pclass+Sex
    """
    df = inputdf.copy()
    df['Age'] = df.groupby(['Sex', 'Pclass'])['Age'].apply(lambda group: group.fillna(group.median()))
    return df

def proc_age_combined(traindf, testdf):
    traindf = traindf.copy()
    testdf = testdf.copy()
    df1 = traindf[['Sex', 'Pclass', 'Age']].copy()
    df2 = testdf[['Sex', 'Pclass', 'Age']].copy()
    df1['source'] = 'train'
    df2['source'] = 'test'
    combined = pd.concat([df1, df2]).reset_index(drop=True)
    combined['Age'] = combined.groupby(['Sex', 'Pclass'])['Age'].apply(lambda group: group.fillna(group.median()))
    traindf['Age'] = combined.loc[combined.source=='train', 'Age']
    testdf['Age']  = combined.loc[combined.source=='test', 'Age']
    return traindf, testdf


train.loc[train.Embarked.isna()]
_ = sns.barplot(x='Embarked', y='Survived', data=train)
train.sort_values('ticketnumber', ascending=False).iloc[424:430]
train
def proc_embarked_trainonly(inputdf):
    df = inputdf.copy()
    df.loc[[829, 61], 'Embarked'] = 'S'
    return df


train.Cabin.unique()
test.Cabin.unique()
t = train.loc[~train.Cabin.isna(), 'Cabin'].apply(lambda x: list([y[0] for y in x.split()])[0])
set(t)
train.loc[train.Cabin.isna()].shape
train['deck'] = 'N'
train.loc[~train.Cabin.isna(), 'deck'] = train.loc[~train.Cabin.isna(), 'Cabin'].apply(lambda x: list([y[0] for y in x.split()])[0])
sns.barplot(x='deck', y='Survived', data=train)
train.groupby('deck').PassengerId.count()
train.loc[train.deck=='T', 'deck'] = 'A'
def proc_cabin(traindf, testdf):
    """
    Extract first letter from Cabin column as 'deck' indicator
    Move most luxerious T cabin as A, as it is a single one.
    """
    traindf = traindf.copy()
    testdf = testdf.copy()
    for df in [traindf, testdf]:
        df['deck'] = 'N'
        df.loc[~df.Cabin.isna(), 'deck'] = df.loc[~df.Cabin.isna(), 'Cabin'].apply(lambda x: list([y[0] for y in x.split()])[0])
        df.loc[df.deck=='T', 'deck'] = 'A'
    return traindf, testdf

s = train.sort_values('Fare').reset_index(drop=True).reset_index()
sns.lineplot(x='index', y='Fare', data=s)
s = test.sort_values('Fare').reset_index(drop=True).reset_index()
sns.lineplot(x='index', y='Fare', data=s)
test.loc[test.Fare.isna()]
pd.concat([train.loc[train.Pclass==3, 'Fare'], test.loc[test.Pclass==3, 'Fare']]).median()
def proc_fare_testonly(inputdf):
    """
    Filling in the single missing value with the median of the Fare of the same Pclass
    """
    df = inputdf.copy()
    df['Fare'] = df['Fare'].fillna(8.05)
    return df


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

def proc_fare_testonly(inputdf):
    """
    Filling in the single missing value with the median of the Fare of the same Pclass
    """
    df = inputdf.copy()
    df['Fare'] = df['Fare'].fillna(8.05)
    return df

def proc_cabin(traindf, testdf):
    """
    Extract first letter from Cabin column as 'deck' indicator
    Move most luxerious T cabin as A, as it is a single one.
    """
    traindf = traindf.copy()
    testdf = testdf.copy()
    for df in [traindf, testdf]:
        df['deck'] = 'N'
        df.loc[~df.Cabin.isna(), 'deck'] = df.loc[~df.Cabin.isna(), 'Cabin'].apply(lambda x: list([y[0] for y in x.split()])[0])
        df.loc[df.deck=='T', 'deck'] = 'A'
    return traindf, testdf

def proc_embarked_trainonly(inputdf):
    """
    There are two missing Embarked values in the train set. Filled it in with S as tickets before and after 
    were also from Southampton (by ticket number)
    """
    df = inputdf.copy()
    df.loc[[829, 61], 'Embarked'] = 'S'
    return df

def proc_age_combined(traindf, testdf):
    """
    Fill in missing ages with median of the same Pclass+Sex groups
    """
    traindf = traindf.copy()
    testdf = testdf.copy()
    df1 = traindf[['Sex', 'Pclass', 'Age']].copy()
    df2 = testdf[['Sex', 'Pclass', 'Age']].copy()
    df1['source'] = 'train'
    df2['source'] = 'test'
    combined = pd.concat([df1, df2]).reset_index(drop=True)
    combined['Age'] = combined.groupby(['Sex', 'Pclass'])['Age'].apply(lambda group: group.fillna(group.median()))
    traindf['Age'] = combined.loc[combined.source=='train', 'Age']
    testdf['Age']  = combined.loc[combined.source=='test', 'Age']
    return traindf, testdf

def proc_tickets(inputdf):
    """
    Split ticket number and prefix
    """
    df = inputdf.copy()
    ticketSplit = df.Ticket.apply(lambda x: x.split()).values
    df['ticketnumber'] = pd.array([int(x[-1]) if x[-1].isdigit() else None for x in ticketSplit], dtype=pd.Int64Dtype())
    df['ticketprefix'] = [np.nan if len(x)==1 else ' '.join(x[:len(x)-1]) for x in ticketSplit]
    return df

def proc_names(inputdf):
    df = inputdf.copy()
    df['title'] = df.Name.str.split().apply(lambda x: [y for y in x if y[-1] =='.'][0])
    df['buyersurname'] = df.Name.apply(lambda x: x.split(',')[0])
    df['surname'] = df.buyersurname # by default it is the same person
    df['maidenname'] = None
    
    boughtForOthers = df.loc[(df.Name.str.contains('\(')) & ~(df.Name.str.contains('\"'))] # bought for others
    # first the none wifes:
    names = boughtForOthers.loc[boughtForOthers.title !='Mrs.'].Name.apply(lambda x: x.split('(')[1].split(')')[0].split()[-1])
    df.loc[names.index, 'surname'] = names
    # now the maiden names of the wifes (this is not perfect and there are some errors)
    names = boughtForOthers.loc[boughtForOthers.title =='Mrs.'].Name.apply(lambda x: x.split('(')[1].split(')')[0])
    names = names.loc[names.apply(lambda x: len(x.split())>1)]
    df.loc[names.index, 'maidenname'] = names
    return df


def importDataset():
    train = pd.read_csv('/kaggle/input/titanic/train.csv')
    test = pd.read_csv('/kaggle/input/titanic/test.csv')
    
    train = proc_names(train)
    test = proc_names(test)
    
    train = proc_tickets(train)
    test = proc_tickets(test)
    
    train, test = proc_age_combined(train, test)
    train, test = proc_cabin(train, test)
    
    train = proc_embarked_trainonly(train)
    test = proc_fare_testonly(test)
    return train, test

train, test = importDataset()

