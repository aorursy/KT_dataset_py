import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import LabelEncoder



train_ds = pd.read_csv('../input/train.csv');

test_ds = pd.read_csv('../input/test.csv');



nonNumeric = ['Name','Sex','Ticket','Cabin','Embarked']



train_ds



# Define some Data-preparation functions

import re

def do_title(d):

    def matchTitle(s):

        out = re.match('.*, (\w+\W*\w*)\. .*?',s)

        if out is None:

            return s;

        else:

            return out.group(1);

        

    #d['Title'] = d.Name.apply(lambda x: x.split(' ')[1]) % too simple, higher error rate

    d['Title'] = d.Name.apply(matchTitle)



do_title(train_ds);

# print(np.unique(train_ds.Title.ravel())) # show the found titles
train_ds

#train_ds[pd.isnull(train_ds.Cabin)]
sns.lmplot(x='NTitle',y='Age',data=train_ds, fit_reg=False);

print(train_ds[train_ds.NTitle ==11].mean())

print(train_ds[train_ds.NTitle ==11].std())
le_title = LabelEncoder()

train_ds['NTitle'] = le_title.fit_transform(train_ds['Title'])



# get inverse transform:

print(np.array([np.array([i,le_title.inverse_transform(x)]) for i,x in enumerate(range(len(le_title.classes_)))]))



le_sex = LabelEncoder();

train_ds['NSex'] = le_sex.fit_transform(train_ds['Sex']);



# get inverse transform:

print(np.array([np.array([i,le_sex.inverse_transform(x)]) for i,x in enumerate(range(len(le_sex.classes_)))]));
train_ds.iloc[:,-2:]


#train_ds[train_ds.Title=='the']
a=train_ds.Age

np.unique(train_ds.Age.ravel())
train_ds[759:760].Name.ravel()