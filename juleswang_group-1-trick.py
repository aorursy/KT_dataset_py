# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



data_path  = "../input/"



people = pd.read_csv(data_path + "people.csv",

                     index_col=0, usecols=["people_id", "group_1"])

train = pd.read_csv(data_path + "act_train.csv",

                    index_col=0, usecols=["people_id", "outcome", "date"])

test = pd.read_csv(data_path + "act_test.csv",

                   index_col=0, usecols=["people_id", "date"])

train = train.join(people)

train = train[["group_1", "date", "outcome"]]

train["date"] = pd.to_datetime(train["date"])

train.sort_values(['group_1', 'date'], inplace=True)

print(train)
c = train.groupby("group_1", sort=False).count()

print(c.describe())

print(c[c.outcome > 10000])
# the group with max activity reccords

group_17304 = train[train['group_1'] == "group 17304"]

print(group_17304.count())

group_17304.plot(ylim=(-1,1), x='date',y='outcome')
group_27940 = train[train['group_1'] == "group 27940"]

print(group_27940.count())

group_27940.plot(ylim=(-2,2), x='date')
def counting_changes(outcome):

    if outcome.shape[0] <= 1:

        return 0

    return np.sum(list(map(lambda x,y:x^y, outcome[1:], outcome[:-1])))



c = train.groupby("group_1", sort=False).agg({'outcome': counting_changes})

print(c.describe())

c.outcome.hist()
# group with change count 3

train[train['group_1'] == "group 12187"].plot(ylim=(-2,2), x='date')
test = test.join(people)

test = test[["group_1", "date"]]

test["date"] = pd.to_datetime(test["date"])

test.sort_values(['group_1', 'date'], inplace=True)

t = test.groupby("group_1", sort=False).count()

test_outcome_change_by_group = t.join(c)



test_outcome_change_by_group['outcome'].loc[np.isnan(test_outcome_change_by_group.outcome)] = -1.01 # to show the nan



test_outcome_change_by_group.outcome.hist()