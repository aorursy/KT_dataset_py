# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

from pandas import DataFrame, Series

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

%matplotlib inline



# machine learning



from sklearn.linear_model import LogisticRegression 

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")



train_df.head(200)



train_df.info()

print('----------------------------')

test_df.info()
x = train_df[[3]]

print (x)
import re

def get_title(name):

    if '.' in name:

        return name.split(',')[1].split('.')[0].strip()

    else:

        return 'Unknown'



# A list with the all the different titles

titles = sorted(set([x for x in train_df.Name.map(lambda x: get_title(x))]))

print('Different titles found on the dataset:')

print(len(titles), ':', titles)

print()



# Normalize the titles, returning 'Mr', 'Master', 'Miss' or 'Mrs'

def replace_titles(x):

    title = x['Title']

    if title in ['Capt', 'Col', 'Don', 'Jonkheer', 'Major', 'Rev', 'Sir']:

        return 'Mr'

    elif title in ['the Countess', 'Mme', 'Lady']:

        return 'Mrs'

    elif title in ['Mlle', 'Ms']:

        return 'Miss'

    elif title =='Dr':

        if x['Sex']=='male':

            return 'Mr'

        else:

            return 'Mrs'

    else:

        return title



# Lets create a new column for the titles

train_df['Title'] = train_df['Name'].map(lambda x: get_title(x))

# train.Title.value_counts()

# train.Title.value_counts().plot(kind='bar')



# And replace the titles, so the are normalized to 'Mr', 'Miss' and 'Mrs'

train_df['Title'] = train_df.apply(replace_titles, axis=1)



# Check that the number of Mr, Mrs and Miss are the same that 'male' and 'female'

print('Title column values. Males and females are the same that for the "Sex" column:')

print(train_df.Title.value_counts())



# Plot the result

train_df.Title.value_counts().plot(kind='bar')

def get_title(name):

    if '.' in name:

        return name.split(',')[1]

    else:

        return 'Unknown'



# A list with the all the different titles

titles = sorted(set([x for x in train_df.Name.map(lambda x: get_title(x))]))

print('Different titles found on the dataset:')

print(len(titles), ':', titles)

print()