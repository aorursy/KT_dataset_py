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
import matplotlib.pyplot as plt
import seaborn as sns 
import warnings
warnings.filterwarnings("ignore")
train = pd.read_csv('../input/train_V2.csv')
train.info()
train.describe()
train.head()
print('Longest kill in a match by a player: ',max(train.longestKill))
print('Longest match duration: ',max(train.matchDuration))
print('How much the marathon runner ran in a match:',max(train.walkDistance))
print('Which match type is played by most players(“solo”, “duo”, “squad”, “solo-fpp”, “duo-fpp”, and “squad-fpp”): ',max(train.matchType))
print('Maximum number of head shots in a match  by a player: ',max(train.headshotKills))
print('Maximum number of kills in a match  by a player: ',max(train.kills))
print('Maximum revivies done in a match  by a player: ',max(train.revives))
print('Maximum distance cover in a vehicle in a match  by a player: ',max(train.rideDistance))
print('Maximum swiming done in a match  by a player: ',max(train.swimDistance))
print('Maximum vehicles destroyed in a match  by a player: ',max(train.vehicleDestroys))
print('Maximum weapons weapons picked up in a match  by a player',max(train.weaponsAcquired))
print('Maximum knock out in a match  by a player',max(train.DBNOs))
print('Maximum consumable items taken in a match  by a player',max(train.boosts))
