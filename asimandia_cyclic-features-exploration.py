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
%%time



import pandas as pd

import numpy as np



# Load data

train = pd.read_csv('../input/cat-in-the-dat/train.csv')

test = pd.read_csv('../input/cat-in-the-dat/test.csv')



print(train.shape)

print(test.shape)
%%time



# Subset

target = train['target']

train_id = train['id']

test_id = test['id']

train.drop(['target', 'id'], axis=1, inplace=True)

test.drop('id', axis=1, inplace=True)



print(train.shape)

print(test.shape)
import seaborn as sns

import matplotlib.pyplot as plt

sns.barplot(train['month'], target)

plt.title("Mean target balance")

f, ax = plt.subplots(1, 2, figsize=(12, 3))

sns.countplot(train['month'], ax=ax[0])

ax[0].set_title("Train")

sns.countplot(test['month'], ax=ax[1])

ax[1].set_title("Test")
import seaborn as sns

import matplotlib.pyplot as plt

sns.barplot(train['day'], target)

plt.title("Mean target balance")

f, ax = plt.subplots(1, 2, figsize=(12, 3))

sns.countplot(train['day'], ax=ax[0])

ax[0].set_title("Train")

sns.countplot(test['day'], ax=ax[1])

ax[1].set_title("Test")