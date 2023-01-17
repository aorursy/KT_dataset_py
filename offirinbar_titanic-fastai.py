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
from fastai import *

from fastai.tabular import *
df_train = pd.read_csv("../input/train.csv")

df_train.head()
df_test = pd.read_csv("../input/test.csv")

#df_test.head()
df_train.describe()
#getting the Title from name

#getting the first letter from Cabin



for df in [df_train, df_test]:

    df['Title'] = df['Name'].str.split(',').str[1].str.split(' ').str[1]

    df['Deck'] = df['Cabin'].str[0]
#concat the to data_frames



all_df = pd.concat([df_train, df_test], sort = False)

all_df.head()
# find mean age for each Title across train and test data sets

mean_age_by_title = all_df.groupby(['Title']).mean()['Age']

mean_age_by_title
# update missing ages

for df in [df_train, df_test]:

    for title, age in mean_age_by_title.iteritems():

        df.loc[df['Age'].isnull() & (df['Title'] == title), 'Age'] = age
df_test.Fare.fillna(0,inplace=True)
df_train.head()
dep_var = 'Survived'

cat_vars = ['Pclass', 'Sex','SibSp','Parch','Embarked','Title','Deck']

cont_vars = ['Age','Fare']

procs = [FillMissing, Categorify, Normalize]
valid_idx = range(len(df_train)-100, len(df_train))

data = (TabularList.from_df(df_train, procs=procs, cont_names=cont_vars, cat_names=cat_vars)

        .split_by_idx(valid_idx)

        .label_from_df(cols=dep_var)

        .add_test(TabularList.from_df(df_test, cat_names=cat_vars, cont_names=cont_vars, procs=procs))

        .databunch())

print(data.train_ds.cont_names)

print(data.train_ds.cat_names)
np.random.seed(101)

learn = tabular_learner(data, layers=[200,100],emb_drop=0.3, metrics=accuracy)
learn.model
data.show_batch(rows=10)
learn.lr_find()

learn.recorder.plot(suggestion = True)
learn.fit_one_cycle(5, 1e-02)
learn.lr_find()

learn.recorder.plot(suggestion = True)
learn.unfreeze()

learn.fit_one_cycle(15, slice(1e-03))
learn.lr_find()

learn.recorder.plot(suggestion = True)
learn.unfreeze()

learn.fit_one_cycle(15, 1e-02)
learn.unfreeze()

learn.fit_one_cycle(200, 1e-02)
predictions, *_ = learn.get_preds(DatasetType.Test)

labels = np.argmax(predictions, 1)
sub_df = pd.DataFrame({'PassengerId': df_test['PassengerId'], 'Survived': labels})

sub_df.to_csv('submission.csv', index=False)
sub_df.tail()