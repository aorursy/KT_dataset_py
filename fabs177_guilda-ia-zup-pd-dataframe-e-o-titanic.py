# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
titanic_train = pd.read_csv('../input/train.csv')
titanic_train.head(5)
len(titanic_train)
titanic_train.columns
titanic_train.dtypes
titanic_train.describe()
import pandas_profiling

pandas_profiling.ProfileReport(titanic_train)
profile = pandas_profiling.ProfileReport(titanic_train)

profile.to_file(outputfile="output.html")
names = titanic_train['Name']

type(names)
names.head()
first_name = titanic_train['Name'][0]

first_name
primeiro_tripulante = titanic_train.loc[0]

primeiro_tripulante
idade = primeiro_tripulante['Age']

idade
df_filter = titanic_train[['Pclass','Parch','Fare']]

df_filter.head(8)
df_ordered_by_age = titanic_train.sort_values(by='Age', ascending=False)

df_ordered_by_age
title_series = titanic_train['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

type(title_series)
titanic_train['Title'] = title_series

titanic_train.head(5)
train_X = titanic_train[['PassengerId', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch',

                        'Ticket', 'Fare', 'Cabin', 'Embarked', 'Title']]

train_Y = titanic_train['Survived']
type(train_X)
type(train_Y)
train_X.tail(7)
train_Y.tail(7)