# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data=pd.read_csv("../input/train.csv")
data.head(10)

#gets the first 10 rows from the dataset
data.tail(10)

#gets the last 10 rows from the dataset
data.shape

#rows,columns
data

#prints the dataset
data.sample(10)

#sample keyword gives the random n rows from the dataframe
data.describe()

#computes summary only for numerical values double/integer
data.info()
total=data.isnull().sum().sort_values(ascending=True)

percent_1=data.isnull().sum()/data.isnull().count()*100

percent_2 = (round(percent_1, 1)).sort_values(ascending=True)

missing_data = pd.concat([total, percent_2], axis=1, keys=['Total', '%'])

missing_data
survived = 'survived'

not_survived = 'not survived'



#fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(10, 4))

women=data[data['Sex']=='female']

men=data[data['Sex']=='male']

ax = sns.distplot(women[women['Survived']==1].Age.dropna(), bins=18, label = survived, kde =False)

ax = sns.distplot(women[women['Survived']==0].Age.dropna(), bins=40, label = not_survived, kde =False)

ax.legend()

ax.set_title('Female')

ax = sns.distplot(men[men['Survived']==1].Age.dropna(), bins=18, label = survived, kde = False)

ax = sns.distplot(men[men['Survived']==0].Age.dropna(), bins=40, label = not_survived, kde = False)

ax.legend()

_ = ax.set_title('Male')