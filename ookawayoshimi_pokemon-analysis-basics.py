# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.patches as patches

%matplotlib inline
# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



df = pd.read_csv('../input/pokemon-data-swordshield/pokemon-swordshield.csv')



print(df.shape)

df.head()
# poke_counts

poke_counts = len(df)

poke_counts
# Basic statistics

df.describe()
df.plot.box()
df.head(25).style.bar(subset=['sum', 'HP', 'Attack', 'Defense', 'Sp_Atk', 'Sp_Def', 'Speed'])
sns.distplot(df["sum"])

plt.show()
# type1

poke_type01 = df['type1'].value_counts()

poke_type01 = pd.DataFrame(poke_type01).T



# type2

poke_type02 = df['type2'].value_counts()

poke_type02 = pd.DataFrame(poke_type02).T



poke_type_join = pd.concat([poke_type01, poke_type02])

poke_type_join
vals1 = [df['type1'].value_counts()[key] for key in df['type1'].value_counts().index]

vals2 = [df['type2'].value_counts()[key] for key in df['type1'].value_counts().index]

inds = np.arange(len(df['type1'].value_counts().index))

width = .45

color1 = np.random.rand(3)

color2 = np.random.rand(3)

handles = [patches.Patch(color=color1, label='type1'), patches.Patch(color=color2, label='type2')]

plt.bar(inds, vals1, width, color=color1)

plt.bar(inds+width, vals2, width, color=color2)

plt.gca().set_xticklabels(df['type1'].value_counts().index)

plt.gca().set_xticks(inds+width)

plt.grid()

plt.xticks(rotation=90)

plt.legend(handles=handles)
poke_type_total = poke_type_join.sum()

poke_type_total



fig, ax = plt.subplots(figsize=(12,6))

data_points = np.arange(len(poke_type_join.columns))



ax.bar(data_points, poke_type_join.iloc[0])



ax.bar(data_points, poke_type_join.iloc[1], bottom=poke_type_join.iloc[0])

ax.set_xticks(data_points)

ax.set_xticklabels(poke_type_join.columns);
percent = poke_type_join.sum()/ poke_counts



missing_data = pd.concat([poke_type_total, percent], axis=1, keys=['Total', 'Percent']).sort_values(by='Total', ascending=False)

missing_data.head(18)
poke_tribe = df[(df['sum'] > 480)]

poke_tribe_list =len(poke_tribe)



print(poke_tribe_list)

print(poke_tribe.shape)
# type1

poke_type480_01 = poke_tribe['type1'].value_counts()

poke_type480_01 = pd.DataFrame(poke_type480_01).T

poke_type480_01



# type2

poke_type480_02 = poke_tribe['type2'].value_counts()

poke_type480_02 = pd.DataFrame(poke_type480_02).T

poke_type480_02



poke_type_480_join = poke_type480_01.append(poke_type480_02)

poke_type_480_join
percent = poke_type_480_join.sum()/ poke_tribe_list



missing_data = pd.concat([poke_type_480_join.sum(), percent], axis=1, keys=['Total', 'Percent']).sort_values(by='Total', ascending=False)

missing_data.head(18)
fig, ax = plt.subplots(figsize=(12,6))

data_points = np.arange(len(poke_type_480_join.columns))



ax.bar(data_points, poke_type_480_join.iloc[0])



ax.bar(data_points, poke_type_480_join.iloc[1], bottom=poke_type_480_join.iloc[0])

ax.set_xticks(data_points)

ax.set_xticklabels(poke_type_480_join.columns);
poke_tribe.sort_values(by='sum', ascending=False) 
poke_tribe.plot.box()
from pandas import plotting

plotting.scatter_matrix(poke_tribe.iloc[:, 2:9], figsize=(8, 8)) 

plt.show()
type1s = list(set(list(df['type1'])))

print(len(type1s), type1s)
poke_tribe_corr = poke_tribe.corr()

corr_math = (poke_tribe_corr.loc[:,['HP','Attack','Defense','Sp_Atk','Sp_Def','Speed']]).corr()

sns.heatmap(corr_math,annot = True)
fig = plt.figure()

ax = fig.add_subplot(111)

ax.violinplot(poke_tribe.iloc[:, 2:8].values.T.tolist())

ax.set_xticks([1, 2, 3, 4, 5, 6, 7]) 

ax.set_xticklabels(poke_tribe.columns[2:8], rotation=90)

plt.grid()

plt.show()
for index, type1 in enumerate(type1s):

    poke_tribe2 = poke_tribe[poke_tribe['type1'] == type1]

    fig = plt.figure(figsize=(8, 4))

    ax = fig.add_subplot(1, 1, 1)

    plt.title(type1)

    ax.set_ylim([0, 280])

    ax.violinplot(poke_tribe2.iloc[:, 2:8].values.T.tolist())

    ax.set_xticks([1, 2, 3, 4, 5, 6]) #データ範囲のどこに目盛りが入るかを指定する

    ax.set_xticklabels(poke_tribe2.columns[2:8], rotation=90)

    plt.grid()

    plt.show()