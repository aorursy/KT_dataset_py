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
train = pd.read_csv('/kaggle/input/pubg-finish-placement-prediction/train_V2.csv')

print('Train shape:', train.shape)
test = pd.read_csv('/kaggle/input/pubg-finish-placement-prediction/test_V2.csv')

print('Test shape:', train.shape)
print(train.head())


# Print train and test columns

print('Train columns:', train.columns.tolist())

print('Test columns:', test.columns.tolist())



# Read the sample submission file

sample_submission = pd.read_csv('/kaggle/input/pubg-finish-placement-prediction/sample_submission_V2.csv')



# Look at the head() of the sample submission

print(sample_submission.head())
train.info()

train.head()

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns 

import warnings

warnings.filterwarnings("ignore")

f,ax = plt.subplots(figsize=(15, 15))

sns.heatmap(train.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
k = 5 #number of variables for heatmap

f,ax = plt.subplots(figsize=(11, 11))

cols = train.corr().nlargest(k, 'winPlacePerc')['winPlacePerc'].index

cm = np.corrcoef(train[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()