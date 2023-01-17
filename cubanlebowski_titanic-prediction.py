# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Import dependencies

%matplotlib inline

# this allows me to see the graphs in place 





# Visualization

import matplotlib.pyplot as plt 

import seaborn as sns

import missingno

plt.style.use('seaborn-whitegrid')



# Preprocessing

from sklearn.preprocessing import OneHotEncoder, LabelEncoder



# Machine Learning

import catboost

from sklearn.model_selection import train_test_split

from sklearn.svm import LinearSVC

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LinearRegression, LogisticRegression, SGDClassifier

from catboost import CatBoostClassifier, Pool, cv



# Ignore warnings

# import the data

train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')

gender_submision = pd.read_csv('../input/titanic/gender_submission.csv')
# Training Data

train.head()





# DATA DESCRIPTION



# PClass => Ticket Class

# SibSP => # of siblings / spouses aboard the Titanic

# Parch => # of parents / children aboard the Titanic

# Embarked => C = Cherbourg, Q = Queenstown, S = Southampton
# Train info

train.info()
# Test Data 

test.head()
test.info()
# Gender Data

gender_submision.head()
# Gender info

# Our submission should look like this

gender_submision.info()
# built in pandas function to describe the data

train.describe()

# Plot graphic of missing values. 

missingno.matrix(train, figsize = (30,5))

missingno.matrix(test, figsize = (30,5))
# A function to show us how many missing values there are

def find_missing_values (df, columns):

#"""Finds number of rows where certain columns are missing values.

#param_df = target dataframe

#param_column = list of columns"""



    missing_vals = {}

    print('Number of missing or NaN values for each column:')

    df_length = len(df)

    for column in columns:

        total_column_values = df[column].value_counts().sum() # this will go thru the columns and total up the missing values

        missing_vals[column] = df_length - total_column_values

        

    return missing_vals



missing_values = find_missing_values(train, columns=train.columns)

missing_values
df_bin = pd.DataFrame() # for discretised continuos variables

df_con = pd.DataFrame() # for continous variables
# What datatype we have?

train.dtypes
# How many people survided?

fig = plt.figure(figsize = (20,2.5))

sns.countplot(y = 'Survived', data = train);

print(train.Survived.value_counts())
df_bin['Survived'] = train['Survived']

df_con['Survived'] = train['Survived']

df_bin.head()
sns.distplot(train.Pclass)

train.Pclass.value_counts()
# How many missing values does Pclass have

missing_values['Pclass']
df_bin['Pclass'] = train['Pclass']

df_con['Pclass'] = train['Pclass']
df_bin.head()
train.Name.value_counts() #value.counts() will always give us teh amount of an element is repeated in a column summed
#df_bin['Name'] = train['Name']

#df_con['Name'] = train['Name']



def sex(x):

    if 'Miss' in x['Name']: return 1

    elif 'Mrs' in x['Name']: return 2

    else: return 3



    

df_bin['Name'] = train.apply(sex, axis=1)

df_con['Name'] = train.apply(sex, axis=1)
df_bin
# Let's view the distribution of sex

plt.figure(figsize = (20,5))

sns.countplot(y='Sex', data = train)
# Check for missing values

missing_values['Sex']
# Adding Sex to the subset dataframe

df_bin['Sex'] = train['Sex']

df_bin['Sex'] = np.where(df_bin['Sex'] == 'female', 0, 1)



df_con['Sex'] = train['Sex']



#df_bin.head()
# Let's check sex compared to survival



fig = plt.figure(figsize = (10,5))

sns.distplot(df_bin.loc[df_bin['Survived'] == 1]['Sex'], kde_kws={'label':'Did not Survived'});

sns.distplot(df_bin.loc[df_bin['Survived'] == 0]['Sex'], kde_kws={'label':'Survived'});
# Let's try to fill the missing values with the avg age of the ones existing.



#missing_age = missing_values['Age'] 

#total_age = train.Age.sum()



#not_missing_age = total_age - missing_age

#not_missing_age



#round(missing_values['Age'] / len(train['Age']), 3)

#not_missing = train.Age.sum() - missing_values['Age'] 

#not_missing



sum_non = (train.Age.sum())

#sum_non

non_miss_cols = len(train['Age']) - missing_values['Age']

avg_age = sum_non / non_miss_cols

avg_age = round(avg_age, 2)

avg_age



# Now let's insert the avg age of the existing values to the missing rows



df_bin['Age'] = train['Age'].fillna(avg_age)

#fd_1 = df_bin[df_bin['Age'] == avg_age]

df_bin

def plot_count_dist(data, bin_df, label_column, target_column, figsize=(20, 5), use_bin_df=False):

    """

    Function to plot counts and distributions of a label variable and 

    target variable side by side.

    ::param_data:: = target dataframe

    ::param_bin_df:: = binned dataframe for countplot

    ::param_label_column:: = binary labelled column

    ::param_target_column:: = column you want to view counts and distributions

    ::param_figsize:: = size of figure (width, height)

    ::param_use_bin_df:: = whether or not to use the bin_df, default False

    """

    if use_bin_df: 

        fig = plt.figure(figsize=figsize)

        plt.subplot(1, 2, 1)

        sns.countplot(y=target_column, data=bin_df);

        plt.subplot(1, 2, 2)

        sns.distplot(data.loc[data[label_column] == 1][target_column], 

                     kde_kws={"label": "Survived"});

        sns.distplot(data.loc[data[label_column] == 0][target_column], 

                     kde_kws={"label": "Did not survive"});

    else:

        fig = plt.figure(figsize=figsize)

        plt.subplot(1, 2, 1)

        sns.countplot(y=target_column, data=data);

        plt.subplot(1, 2, 2)

        sns.distplot(data.loc[data[label_column] == 1][target_column], 

                     kde_kws={"label": "Survived"});

        sns.distplot(data.loc[data[label_column] == 0][target_column], 

                     kde_kws={"label": "Did not survive"});
#Missing values

missing_values['SibSp']
train.SibSp.value_counts()
df_bin['SibSp'] = train['SibSp']

df_con['SibSp'] = train['SibSp']



df_bin
# Visualize the counts of SibSp and the distributions of the values against survived

plot_count_dist(train,

               bin_df=df_bin,

               label_column='Survived',

               target_column='SibSp',

               figsize=(20,10))
train.Parch.value_counts()
missing_values['Parch']
df_bin['Parch'] = train['Parch']

df_con['Parch'] = train['Parch']
# Visualize the count of Parch and the distribution of teh values

# against survived



plot_count_dist(data = train, 

                bin_df=df_bin,

                label_column='Survived', 

                target_column='Parch', 

                figsize=(20, 10))
# How many kinds of tickets are there

missing_values['Ticket']
train.Ticket.value_counts()
sns.countplot(y='Ticket', data=train);
missing_values['Fare']
sns.countplot(y='Fare', data=train);
train.Fare.dtype
# How many fares do we have out there

print('There are', len(train.Fare.unique()), 'unique Fare values')
df_con['Fare'] = train['Fare'] 

df_bin['Fare'] = pd.cut(train['Fare'],4) #discretized
df_bin.head()

df_con.head()

df_bin.Fare.value_counts()
# Visualize the Fare bin counts as well as the fare distribution versus survived

# info is splitted in 4 bins for a better visualization... 4 Bins so we can have data in all bins otherwise we get 1 empty bin with 5

plot_count_dist(data = train, 

                bin_df=df_bin,

                label_column='Survived', 

                target_column='Fare', 

                figsize=(20, 10),

                use_bin_df=True)
train.Fare.value_counts(normalize=True, sort = True)

train.Fare.min()
missing_values['Cabin']
train.Cabin.value_counts()
#Let's copy the same value for Embarked to make it to cabin

df_bin['Cabin'] = df_bin['Embarked']

df_con['Cabin'] = df_con['Embarked']
df_bin.head()
train.info()
missing_values['Embarked']
train['Embarked']
def embarked(x):

    if x['Embarked'] == 'S': return 'A'

    elif x['Embarked'] == 'Q': return 'B'

    else : return 'C'

    

    print (x)

    

df_bin['Embarked'] = train.apply(embarked, axis=1)

df_con['Embarked'] = train.apply(embarked, axis=1)
df_bin

train[train.Embarked.isnull()]
df_bin
# What those counts look like 

sns.countplot(y='Embarked', data=train);
train.Embarked.value_counts()
#train[train.Embarked.isnull()]

train = train.dropna(subset=['Embarked'])

train.info