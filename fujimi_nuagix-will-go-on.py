# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pandas import Series,DataFrame

import matplotlib.pyplot as plt

import seaborn as sns

import re as re

sns.set_style('whitegrid')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

train_data = pd.read_csv("../input/train.csv")

test_data  = pd.read_csv("../input/test.csv")

train_data.head()
train_data = train_data.drop(['PassengerId'], axis=1)

train_data.head()
print (train_data[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean())
print (train_data[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean())
print (train_data[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean())
full_data = [train_data, test_data]

for dataset in full_data:

    dataset['FamillySize'] = dataset['SibSp'] + dataset['Parch'] + 1

print (train_data[['FamillySize', 'Survived']].groupby(['FamillySize'], as_index=False).mean())
avg_age = train_data['Age'].mean()

std_age = train_data['Age'].std()

print ("avg age : ", avg_age)

print ("std age : ", std_age)
def get_title (name):

    title_search = re.search(" ([A-Za-z]+)\.", name)

    if title_search:

        return title_search.group(1)

    return ""

for dataset in full_data:

    dataset['Title'] = dataset['Name'].apply(get_title)

print (pd.crosstab(train_data['Title'], train_data['Sex']))

print (train_data.loc[train_data['Title']=='Jonkheer'])

train_data.describe()
train_data.describe(include=['O'])