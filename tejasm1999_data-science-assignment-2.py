# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import missingno

import seaborn as sns

plt.style.use('seaborn-whitegrid')

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('/kaggle/input/titanicdataset-traincsv/train.csv')
train.head(15)
train.Age.plot.hist()
train.describe()
missingno.matrix(train, figsize = (30,10))
train.isnull().sum()
df_bin = pd.DataFrame() 

df_con = pd.DataFrame()
train.dtypes
fig = plt.figure(figsize=(20,1))

sns.countplot(y='Survived', data=train);

print(train.Survived.value_counts())
df_bin['Survived'] = train['Survived']

df_con['Survived'] = train['Survived']
df_bin.head()
df_con.head()
sns.distplot(train.Pclass)
train.Pclass.isnull().sum()
df_bin['Pclass'] = train['Pclass']

df_con['Pclass'] = train['Pclass']
train.Name.value_counts()
plt.figure(figsize=(20, 5))

sns.countplot(y="Sex", data=train);
train.Sex.isnull().sum()
train.Sex.head()
train.Age.isnull().sum()
def plot_count_dist(data, bin_df, label_column, target_column, figsize=(20, 5), use_bin_df=False):

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
train.SibSp.isnull().sum()
train.SibSp.value_counts()
df_bin['SibSp'] = train['SibSp']

df_con['SibSp'] = train['SibSp']
plot_count_dist(train, 

                bin_df=df_bin, 

                label_column='Survived', 

                target_column='SibSp', 

                figsize=(20, 10))
train.Parch.isnull().sum()
train.Parch.value_counts()
df_bin['Parch'] = train['Parch']

df_con['Parch'] = train['Parch']
train.Ticket.isnull().sum()
sns.countplot(y="Ticket", data=train);
train.Ticket.value_counts()
train.Fare.isnull().sum()
sns.countplot(y="Fare", data=train);
df_con['Fare'] = train['Fare'] 

df_bin['Fare'] = pd.cut(train['Fare'], bins=5)
plot_count_dist(data=train,

                bin_df=df_bin,

                label_column='Survived', 

                target_column='Fare', 

                figsize=(20,10), 

                use_bin_df=True)

train.Cabin.isnull().sum()
train.Cabin.value_counts()
train.Embarked.isnull().sum()
train.Embarked.value_counts()
sns.countplot(y='Embarked', data=train);