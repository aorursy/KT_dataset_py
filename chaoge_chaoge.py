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
import csv as csv

from sklearn.ensemble import RandomForestClassifier
train_df = pd.read_csv('../input/train.csv',header=0)
train_df['Gender'] = train_df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

if len(train_df.Embarked[train_df.Embarked.isnull()]) > 0:

    train_df.Embarked[train_df.Embarked.isnull()] = train_df.Embarked.dropna().mode().values

Ports = list(enumerate(pd.unique(train_df['Embarked'])))

Ports_dict = {name: i for i, name in Ports}

train_df.Embarked = train_df.Embarked.map(lambda x: Ports_dict[x]).astype(int)

median_age = train_df['Age'].dropna().median()

if len(train_df.Age[train_df.Age.isnull()]) > 0:

    train_df.loc[(train_df.Age.isnull()), 'Age'] = median_age

train_df = train_df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId'], axis=1)

test_df['Gender'] = test_df['Sex'].map({'female': 0, 'male': 1}).astype(int)
