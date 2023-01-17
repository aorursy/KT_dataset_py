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
data = pd.read_csv('../input/train_V2.csv')
data.loc[:5,]
data_squad_fpp = data.loc[data.matchType=='squad-fpp']

print(data_squad_fpp.groupId.value_counts().describe())

data_squad_fpp.groupId.value_counts().plot(kind='box')

data_squad = data.loc[data.matchType=='squad']

print(data_squad.groupId.value_counts().describe())

data_squad.groupId.value_counts().plot(kind='box')

data_duo_fpp = data.loc[data.matchType=='duo-fpp']

print(data_duo_fpp.groupId.value_counts().describe())

data_duo_fpp.groupId.value_counts().plot(kind='box')

data_duo = data.loc[data.matchType=='duo']

print(data_duo.groupId.value_counts().describe())

data_duo.groupId.value_counts().plot(kind='box')

data_solo_fpp = data.loc[data.matchType=='solo-fpp']

print(data_solo_fpp.groupId.value_counts().describe())

data_solo_fpp.groupId.value_counts().plot(kind='box')

data_solo = data.loc[data.matchType=='solo']

print(data_solo.groupId.value_counts().describe())

data_solo.groupId.value_counts().plot(kind='box')

data_others = data.loc[(data.matchType!='solo')|(data.matchType!='solo-fpp')|(data.matchType!='duo')|(data.matchType!='duo-fpp')|(data.matchType!='squad')|(data.matchType!='squad-fpp')]

print(data_others.groupId.value_counts().describe())

data_others.groupId.value_counts().plot(kind='box')

data_match_group_player = data.groupby('matchId').groupId.nunique()
data_match_group_player
data_match_group_player.describe()

data_match_group_player.plot(kind='box')

print('The median number of groups per match is {}.'.format(data_match_group_player.median()))
data.groupby('groupId').matchId.nunique().sort_values(ascending=False)
data_continuous = data.loc[:,[data.columns[i] for i, x in enumerate(data.dtypes) if x != 'object']]
data_continuous.iloc[:10]
import seaborn as sns
import matplotlib.pyplot as plt

data_continuous_corr = data_continuous.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(data_continuous_corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 11))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(data_continuous_corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
data_continuous_corr.loc['winPlacePerc',:].sort_values()

print('Variables with correlation greater than 0.5, or less than -0.5, with winPlacePerc are {}.'.format([data_continuous_corr.index[i] for i, x in enumerate(data_continuous_corr.loc['winPlacePerc',:]) if abs(x) > 0.5 ]))
high_corr_var = {}
for x in data_continuous_corr.index:
    for y in data_continuous_corr.columns:
        if abs(data_continuous_corr.loc[x,y]) > 0.5:
            high_corr_var[(x,y)] = data_continuous_corr.loc[x,y]
pd.DataFrame([x for x in high_corr_var.values()], index=high_corr_var.keys(), columns = ['Correlation'])
print('When winPoints = 0, killPoints are also {}.'.format(
    sum(data_continuous.loc[data_continuous.winPoints==0, 'killPoints'])))

print('When winPoints = 0, the minimum rankPoints is {}.'.format(
    min(data_continuous.loc[data_continuous.winPoints==0, 'rankPoints'])))
print('When killPoints = 0, winPoints are also {}.'.format(
    sum(data_continuous.loc[data_continuous.killPoints==0, 'winPoints'])))

print('When killPoints = 0, the minimum rankPoints is {}.'.format(
    min(data_continuous.loc[data_continuous.killPoints==0, 'rankPoints'])))
print('When rankPoints = -1, the minimum winPoints and killPoints is {} and {} respectively.'.format(
    min(data_continuous.loc[data_continuous.rankPoints==-1, 'winPoints']),
    min(data_continuous.loc[data_continuous.rankPoints==-1, 'killPoints'])
))
print('When rankPoints > 0, the minimum winPoints and killPoints is {} and {} respectively.'.format(
    min(data_continuous.loc[data_continuous.rankPoints>0, 'winPoints']),
    min(data_continuous.loc[data_continuous.rankPoints>0, 'killPoints'])
))
