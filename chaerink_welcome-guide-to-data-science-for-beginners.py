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
# The classic visualization tool in Python. Old, but still powerful.

import matplotlib.pyplot as plt 

# A easy-to-use tool based on matplotlib

import seaborn as sns

# A handy library for missing value detection

import missingno as msno
train = pd.read_csv('/kaggle/input/pubg-finish-placement-prediction/train_V2.csv')

test = pd.read_csv('/kaggle/input/pubg-finish-placement-prediction/test_V2.csv')

train.head()
print("Number of Columns: {}".format(len(train.columns)))

train.columns
train.dtypes.sort_values(ascending=False)
[x for x in train.columns if x not in test.columns]
msno.matrix(train)
train.isna().sum()
train[np.isnan(train['winPlacePerc'])]
train[train['matchId']=='224a123c53e008']
train.dropna(inplace=True)
def dist_plot(col, data=train):

    plt.figure(figsize=(10,6.5))

    sns.distplot(data[col])

    plt.title("Distribution_{}".format(col))
dist_plot('winPlacePerc')



# Rather evenly distributed, as it ought to be.
dist_plot('kills')
dist_plot('assists')
dist_plot('heals')
dist_plot('damageDealt')
dist_plot('walkDistance')
train['kills'].describe(percentiles=[0.1*x for x in range(10)])

# This prints the threshold value for each 10th percentile
temp = train.groupby('kills')['winPlacePerc'].mean()



plt.figure(figsize=(10, 7))

plt.bar(temp.index, temp, color='peachpuff')

plt.plot(temp, color='chocolate')

plt.plot(temp.index, np.ones(len(temp)), ls='--', alpha=0.5, color='k')

plt.title("Kills and Win Place")

plt.xlabel("# Kills")

plt.ylabel("Win Place")
plt.figure(figsize=(12,8))

plt.scatter(train['kills'], train['winPlacePerc'], s=1, color='plum')

plt.title('kills-winPlace')