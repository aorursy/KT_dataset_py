# import Libraries 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # plotting

import seaborn as sns # combined plotting 

df = pd.read_csv('../input/train.csv', index_col=0)

df_test = pd.read_csv('../input/test.csv', index_col=0)
df.head(5)
df.dtypes
df.Survived.hist()
## Function to plot conditioned histograms

def cond_hists(df, plot_cols, grid_col):

    

    ## Loop over the list of columns

    for col in plot_cols:

        grid1 = sns.FacetGrid(df, col=grid_col)

        grid1.map(plt.hist, col, alpha=.7, bins=6)

    return grid_col



## we need to conver the Sex and Embarked colums to numbers for the function to work 

df.Sex = df.Sex.map({'male':0, 'female':1})

df_test.Sex = df_test.Sex.map({'male':0, 'female':1})



## Define columns for making a conditioned histogram

plot_cols = ["Pclass",

               "Age",

               "Sex",

               "Parch",

               "SibSp"]



cond_hists(df, plot_cols, 'Survived')
df.corr()
df.hist(column='Survived', by='Embarked', bins =3)
df.hist(column='Pclass', by='Embarked', bins=5)
df.boxplot(column='Fare', by='Survived')
df.boxplot(column='Fare', by='Pclass')
df.boxplot(column='Fare', by='Parch')
df.isnull().sum()
df.Embarked = df.Embarked.fillna('S')
# fill NAN first with something, and replace all

df.Cabin = df.Cabin.fillna('U')

df.Cabin = df.Cabin.map(lambda x: 0 if x == 'U' else 1)



# repeat for the test dataset 

df_test.Cabin = df.Cabin.fillna('U')

df_test.Cabin = df.Cabin.map(lambda x: 0 if x == 'U' else 1)
df['Title'] = df.Name.map(lambda x: x.split(',')[1].split('.')[0])

df_test['Title'] = df_test.Name.map(lambda x: x.split(',')[1].split('.')[0])

df.Title.unique()
df.Title = df.Title.map({' Mr':0, ' Mrs':1, ' Miss':2, ' Master':3, ' Don':4, ' Rev':5, ' Dr':6, ' Mme':7, ' Ms':8, ' Major':9, ' Lady':10, ' Sir':11, ' Mlle':12, ' Col':13, ' Capt':14, ' the Countess':15, ' Jonkheer':16})

df.boxplot(column='Age', by='Title')
df.corr()
age1 = df[(df.Pclass == 1) & (df.Age.isnull() == False)]['Age'].mean()

age2 = df[(df.Pclass == 2) & (df.Age.isnull() == False)]['Age'].mean()

age3 = df[(df.Pclass == 3) & (df.Age.isnull() == False)]['Age'].mean()



df.loc[(df.Pclass == 1) & (df.Age.isnull() == True),'Age'] = age1

df.loc[(df.Pclass == 2) & (df.Age.isnull() == True),'Age'] = age2

df.loc[(df.Pclass == 3) & (df.Age.isnull() == True),'Age'] = age3



df_test.loc[(df_test.Pclass == 1) & (df_test.Age.isnull() == True),'Age'] = age1

df_test.loc[(df_test.Pclass == 2) & (df_test.Age.isnull() == True),'Age'] = age2

df_test.loc[(df_test.Pclass == 3) & (df_test.Age.isnull() == True),'Age'] = age3
df.hist(column='Survived', by='SibSp', bins = 3)
def cond_SibSp (x):

    if x == 0:

        return 0 

    if x in [1,2]:

        return 1

    if x > 2:

        return 2

    

df.SibSp = df.SibSp.map(cond_SibSp)

df_test.SibSp = df_test.SibSp.map(cond_SibSp)

df.hist(column='SibSp', by='Survived', bins = 3)
df.hist(column='Survived', by='Parch', bins = 3)
df.Parch = df.Parch.map(lambda x: 0 if x == 0 else 1)

df_test.Parch = df_test.Parch.map(lambda x: 0 if x == 0 else 1)
df.hist(column='Survived', by='Parch', bins = 3)
df.corr()