# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv') #loading in the training set
print(train.head()) #examining the first few rows of the training set
train
train.isnull().values.any()
list(train)
import seaborn as sns

#features 1: Fighting
f1 = (
    train.loc[:, ["winPlacePerc","killPoints", "killPlace", "kills", "DBNOs",  "headshotKills", "assists", "longestKill","heals", "revives" , "teamKills" ]]
).corr()
plt.pyplot.figure(figsize=(14, 12))
sns.heatmap(f1, annot = True)
#teamKills are Friendly Fire kills:
train["teamKills"].describe()
###########
#features 2: Other data
f2 = (
    train.loc[:, ["winPlacePerc","winPoints", "maxPlace", "numGroups", "walkDistance", "swimDistance" , "rideDistance", "roadKills", 'vehicleDestroys', "weaponsAcquired" ]]
).corr()
plt.pyplot.figure(figsize=(12, 10))
sns.heatmap(f2, annot = True)