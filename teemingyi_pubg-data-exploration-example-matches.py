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
data_solo_example = data.loc[data.matchId == data.loc[data.matchType=='solo','matchId'].iloc[0],:].sort_values(by='winPlacePerc',ascending=False)
print(data_solo_example)

print('This example has {} players.'.format(len(data_solo_example.Id)))
print('This example has {} groups.'.format(len(data_solo_example.groupId.unique())))

[x-data_solo_example.winPlacePerc.iloc[i+1] for i, x in enumerate(data_solo_example.winPlacePerc) if i < len(data_solo_example.winPlacePerc)-1]
data_solo_example.loc[data_solo_example.groupId.isin(
    list(data_solo_example.groupId.value_counts().loc[data_solo_example.groupId.value_counts()>1].index
        ))]
data_duo_example = data.loc[data.matchId == data.loc[data.matchType=='duo','matchId'].iloc[0],:].sort_values(by='winPlacePerc',ascending=False)
print(data_duo_example)

print('This example has {} players.'.format(len(data_duo_example.Id)))
print('This example has {} groups.'.format(len(data_duo_example.groupId.unique())))

data_duo_example.maxPlace
data_duo_example.loc[data_duo_example.groupId.isin(
    list(data_duo_example.groupId.value_counts().loc[data_duo_example.groupId.value_counts()>2].index
        ))]
data_squad_example = data.loc[data.matchId == data.loc[data.matchType=='squad','matchId'].iloc[0],:].sort_values(by='winPlacePerc',ascending=False)
print(data_squad_example)

print('This example has {} players.'.format(len(data_squad_example.Id)))
print('This example has {} groups.'.format(len(data_squad_example.groupId.unique())))

data_squad_example.maxPlace
data_squad_example.loc[data_squad_example.groupId.isin(
    list(data_squad_example.groupId.value_counts().loc[data_squad_example.groupId.value_counts()>4].index
        ))]
