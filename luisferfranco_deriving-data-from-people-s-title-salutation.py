import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import re



from sklearn import tree

from sklearn.ensemble import RandomForestClassifier



%matplotlib inline
train = pd.read_csv("../input/train.csv")

test  = pd.read_csv("../input/test.csv")
train.describe()
test.describe()
test['Fare'] = test['Fare'].fillna(test['Fare'].mean())
train['Age1'] = train['Age']

test['Age1']  = test['Age']



# I'm keeping the mean to impute the exact same value to the test dataframe

agemean = train['Age1'].mean()



train['Age1'] = train['Age1'].fillna(agemean)

test['Age1']  = test['Age1'].fillna(agemean)

train[['Age', 'Age1']][5:6]
train['Age1'].hist()
test['Name'].head()
titlere = re.compile(r'\w+,\s(\w+.)*')
def setTitle(df):

    df['Title'] = df['Name']

    df['Title'] = df['Title'].apply(lambda t: titlere.search(t).group(1))
setTitle(train)

setTitle(test)
train[['Name', 'Title']].head()
train.boxplot('Age', by='Title', rot=90)
def setAge2(df):

    df['Age2'] = df['Age']

    for i in df['Title'].value_counts().keys():

        df['Age2'][df['Title'] == i] = df['Age2'][df['Title'] == i].fillna(df['Age2'][df['Title'] == i].mean())

    df['Age2'] = df['Age2'].fillna(df['Age2'].mean())
setAge2(train)

setAge2(test)
train[['Age1', 'Age2']].hist(sharey=True)
test[['Age1', 'Age2']].hist(sharey=True)