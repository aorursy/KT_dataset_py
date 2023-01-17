# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")

test = pd.read_csv("../input/titanic/test.csv")

train = pd.read_csv("../input/titanic/train.csv")
train.shape # Rows -891, Columns - 12


train.columns
for i in train.columns:

    print(i)
# To check which are all the columns having numerical values

num=train.median().index
num
# to check which are all the columns having categorical values (except numerical colums)

cat=[i for i in train.columns if i not in num]
cat
# length of "categorical" & "numerical" should match with lenght of dataset columns

len(cat),len(num),len(train.columns)
# To check null values columns preset or not in the given dataset if yes what is the count of each columns

train.isna().sum()
# to check percentage of the null values present in the each columns inthe dataset

nullvalues=(train.isna().sum()/train.shape[0])*100
nullvalues
# Based on client input you can drop the null values(if client says consider the null values if greate than 50%)

# Below are the command to drop the null values based on client input

# train.fillna(20,inplace=True) - beginer can use this code to treat Null value

drop_columns=nullvalues[nullvalues>int(input())].index

retain_columns=nullvalues[nullvalues<=int(input())].index
# lengh of drop columns and length of retain columns should match

len(drop_columns),len(retain_columns)
# Replace the null values in numerical columns with "median"

# Replace the null values in categorical cloumns with "average counts"

for i in cat:

    train[i].fillna(train[i].value_counts().index[0],inplace=True)

for i in num:

    train[i].fillna(train[i].median(),inplace=True)
# Finally to check if any null values present or not 

# Null values treatment done and you can see all the columns having zero null values 

train.isna().sum()