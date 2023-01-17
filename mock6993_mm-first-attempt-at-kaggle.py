# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

# data analysis and wrangling

import pandas as pd

import numpy as np

import random as rnd



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier





# Any results you write to the current directory are saved as output.


test = pd.read_csv('../input/test.csv')

df = pd.read_csv('../input/train.csv')
df.info()
df = df.drop(['Ticket', 'Cabin'], axis=1)
def name_extract(word):

    return word.split(',')[1].split('.')[0].strip()
df2 = pd.DataFrame({'Salutation':df['Name'].apply(name_extract)})

df = pd.merge(df, df2, left_index = True, right_index = True) # merges on index

temp1 = df.groupby('Salutation').PassengerId.count()

print(temp1)

def group_sal(old_sal):

    if old_sal == 'Mr':

        return('Mr.')

    else:

        if old_sal == 'Mrs':

            return('Mrs.')

        else:

            if old_sal == 'Miss':

                return('Miss')

            else:

                if old_sal == 'Master':

                    return('Master')

                else:

                    return("Others")
df3 = pd.DataFrame({'New_Salutation':df['Salutation'].apply(group_sal)})

df = pd.merge(df, df3, left_index = True, right_index = True)

temp1 = df3.groupby('New_Salutation').count()

temp1.head()

df.boxplot(column='Age', by = 'New_Salutation')
table = df.pivot_table(values='Age',index=['New_Salutation'],columns=['Pclass', 'Sex'], aggfunc=np.median)

#define function to return value of this pivot table

def fage(x):

    return table[x['Pclass']][x['Sex']][x['New_Salutation']]

df['Age'].fillna(df[df['Age'].isnull()].apply(fage, axis=1), inplace=True)

df.head()
df = df.dropna(axis=0, how='any')
df.loc[(df.Fare >= 200) & (df.Pclass == 1), 'Fare'] = 60
sur = df.Survived.sum()

perc = round(((sur/(df.Survived.count())) *100),2)

print(str(perc) + '% ' + 'survived from this sample')
df.describe()
#pivot table showing survival rates by class

df[['Pclass','Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
df[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
df[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
df[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
h = sns.FacetGrid(df, col='Survived')

h.map(plt.hist, 'Age', bins=20)