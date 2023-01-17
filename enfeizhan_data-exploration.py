# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
import seaborn as sns
import matplotlib.pyplot as plt

# Any results you write to the current directory are saved as output.
stats = pd.read_csv('../input/stats.csv')
stats.head()
stats.shape
stats.isnull().sum()
stats.Team.nunique()
stats.Season.unique()
stats.groupby('Season')['Round'].nunique()
def round_mapping(round_str):
    if round_str[0] == 'R':
        return int(round_str[1:])
    elif round_str == 'EF' or round_str == 'QF':
        return 24
    elif round_str == 'SF':
        return 25
    elif round_str == 'PF':
        return 26
    elif round_str == 'GF':
        return 27
stats.loc[:, 'Round in Number'] = stats.Round.apply(round_mapping)
stats.set_index('Round in Number').loc[(stats.Season == 2015) & (stats.Player == 'Atkins, Rory'), 'Disposals']
fig, ax = plt.subplots(figsize=(15, 8))
stats.loc[(stats.Season == 2015) & (stats.Player == 'Atkins, Rory'), ['Round in Number', 'Disposals']].set_index('Round in Number').loc[:, 'Disposals'].plot(ax=ax, label='2015');
stats.loc[(stats.Season == 2016) & (stats.Player == 'Atkins, Rory'), ['Round in Number', 'Disposals']].set_index('Round in Number').loc[:, 'Disposals'].plot(ax=ax, label='2016');
ax.grid()
ax.legend();
