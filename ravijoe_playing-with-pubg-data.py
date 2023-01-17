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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns 

import warnings

warnings.filterwarnings("ignore")
sample_submission_V2 = pd.read_csv("../input/pubg-finish-placement-prediction/sample_submission_V2.csv")

test = pd.read_csv("../input/pubg-finish-placement-prediction/test_V2.csv")

train = pd.read_csv("../input/pubg-finish-placement-prediction/train_V2.csv")
train.info()

train.describe()
train.head()

train.headshotKills.unique()


# Bar plot

sns.countplot(x='headshotKills', data=train)

plt.show()



# Histogram

sns.distplot(train['headshotKills'])

plt.show()
print("The average person kills {:.4f} players, 99% of people have {} kills or less, while the most kills ever recorded is {}.".format(train['kills'].mean(),train['kills'].quantile(0.99), train['kills'].max()))
# data = train.copy()

# data.loc[data['kills'] > data['kills'].quantile(0.99)] = '8+'

data=train.kills.value_counts()

plt.figure(figsize=(15,10))

sns.distplot(data)

plt.title("Kill Count",fontsize=15)

plt.show()
data = train.copy()

data = data[data['kills']==0]

plt.figure(figsize=(15,10))

plt.title("Damage Dealt by 0 killers",fontsize=15)

sns.distplot(data['damageDealt'])

plt.show()
print("{} players ({:.4f}%) have won without a single kill!".format(len(data[data['winPlacePerc']==1]), 100*len(data[data['winPlacePerc']==1])/len(train)))
import matplotlib.pyplot as plt

from scipy import stats

plt.figure(figsize=(15,10))

sns.distplot(train.damageDealt, bins=20,kde=False)

plt.title("Damage Dealt by all killers",fontsize=15)

plt.show()

import matplotlib.pyplot as plt

from scipy import stats

plt.figure(figsize=(15,10))

sns.distplot(train.kills, bins=20,kde=False)

plt.title("Kills",fontsize=15)

plt.show()
import matplotlib.pyplot as plt

plt.figure(figsize=(15,10))

sns.jointplot(x="damageDealt", y="kills", data=train);

plt.title("Kills",fontsize=15)

plt.show()
sns.jointplot(x="winPlacePerc", y="kills", data=train)

plt.show()
sns.jointplot(x="heals", y="kills", data=train)

plt.show()


import matplotlib.pyplot as plt

plt.figure(figsize=(25,20))

sns.heatmap(train.corr())

plt.show()
