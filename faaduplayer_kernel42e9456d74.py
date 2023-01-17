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
# Disabling warnings

import warnings

warnings.simplefilter("ignore")



#importig visualization libraries

import matplotlib.pyplot as plt

import seaborn as sns
#set the dataframes

train = pd.read_csv("../input/titanic/train.csv")

test = pd.read_csv("../input/titanic/test.csv")

sub = pd.read_csv("../input/titanic/gender_submission.csv")
train.head()
test.head()
sub.head()
#data size

print("Train data size - ", train.shape)

print("Test data size - ", test.shape)
#check for nulls

train.isnull().sum()
test.isnull().sum()
from sklearn.impute import SimpleImputer



imputer = SimpleImputer(np.nan, "mean")



train['Age'] = imputer.fit_transform(np.array(train['Age']).reshape(891, 1))

train.drop(['PassengerId', 'Name', 'Cabin'], axis=1, inplace=True)



test['Age'] = imputer.fit_transform(np.array(test['Age']).reshape(418, 1))

test.drop(['PassengerId', 'Name', 'Cabin'], axis=1, inplace=True)
train.head()
test.head()