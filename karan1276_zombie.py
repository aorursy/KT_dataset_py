Zombie# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# visualization

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/train.csv")

combine = [train_df, test_df]

train_df.head()
train_df.info()

train_df.describe()
sns.countplot(x="OutcomeType", hue="SexuponOutcome", data=train_df)
g = sns.FacetGrid(train_df, row='SexuponOutcome', col='AnimalType')

g.map(sns.countplot, "OutcomeType")
print("Before", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)



train_df = train_df.drop('Name', axis=1)

test_df = test_df.drop('Name', axis=1)

combine = [train_df, test_df]



print("After", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)
print("Before", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)



for dataset in combine:

    dataset['Year'] = dataset.DateTime.str.extract('([0-9]{4})', expand=False)



#Now we drop DateTime

train_df = train_df.drop('DateTime', axis=1)

test_df = test_df.drop('DateTime', axis=1)

combine = [train_df, test_df]



print("After", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)
train_df.head()
sns.heatmap(train_df.corr())