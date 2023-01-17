# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



#Libs

import numpy as np

import pandas as pd

from scipy import stats

import seaborn as sns

import matplotlib.pyplot as plt

pd.set_option('max_columns', None)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

dir =[]

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        dir.append(os.path.join(dirname, filename))

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
per_game_stats = pd.read_csv('/kaggle/input/michael-jordan-kobe-bryant-and-lebron-james-stats/per_game_stats.csv')

per_game_stats.head()
# MP

def compare_MP():

    subset = per_game_stats.loc[:, ['Player','MP']].groupby('Player').median()

    return subset



compare_MP()
#3p%

def compare_3P():

    subset = per_game_stats.loc[:, ['Player','3P%']].groupby('Player').median()

    return subset



compare_3P()
#2p%

def compare_2P():

    subset = per_game_stats.loc[:, ['Player','2P%']].groupby('Player').median()

    return subset



compare_2P()
#blks

def compare_BLK():

    subset = per_game_stats.loc[:, ['Player','BLK']].groupby('Player').median()

    return subset



compare_BLK()
#ptskobe

def compare_PTS():

    subset = per_game_stats.loc[:, ['Player','PTS']].groupby('Player').median()

    return subset



compare_PTS()
#Pts for MPs



def compare_PTS_MP():



    subset = per_game_stats.loc[:, ['Player','PTS', 'MP']]

    subset['PTS/MP'] = subset['PTS'] / subset['MP']

    subset = subset[['Player','PTS/MP']].groupby('Player').median()

    return subset



compare_PTS_MP()
MJ = per_game_stats.loc[per_game_stats['Player'] == 'Michael Jordan', ['PTS']]

print(f'Tamanho da amostra {MJ.shape}')



LBJ = per_game_stats.loc[per_game_stats['Player'] == 'Lebron James', ['PTS']]

print(f'Tamanho da amostra {LBJ.shape}')



kobe = per_game_stats.loc[per_game_stats['Player'] == 'Kobe Bryant', ['PTS']]

print(f'Tamanho da amostra {kobe.shape}')
# Checking the distributions

plt.figure(figsize = (12,8))

sns.distplot(MJ, label='MJ')

sns.distplot(LBJ, label='LBJ')

sns.distplot(kobe, label='Kobe')

plt.legend()
f, (ax,ax1,ax2) = plt.subplots(1,3, figsize=(12,8))

ax.boxplot(MJ['PTS'])

ax1.boxplot(LBJ['PTS'])

ax2.boxplot(kobe['PTS'])
# homoscedasticity - To determine if the variations are different.



def check_homoscedasticity(obj1, obj2, obj3):

    return stats.levene(obj1['PTS'].tolist(),obj2['PTS'].tolist(),obj3['PTS'].tolist())



check_homoscedasticity(MJ, LBJ, kobe)
# H-test (kruskal)



def Kruskal_ANOVA(obj1, obj2, obj3):

    anova = stats.kruskal(obj1, obj2, obj3)

    return anova



Kruskal_ANOVA(MJ, LBJ, kobe)
# Data

subset = per_game_stats.loc[:, ['Player','PTS', 'MP']]

subset['PTS/MP'] = subset['PTS'] / subset['MP']
# Checking the distributions



subset_MJ = subset.loc[subset['Player'] == 'Michael Jordan',['PTS/MP'] ]

subset_LBJ = subset.loc[subset['Player'] == 'Lebron James',['PTS/MP'] ]

subset_kobe = subset.loc[subset['Player'] == 'Kobe Bryant',['PTS/MP'] ]



plt.figure(figsize = (12,8))

sns.distplot(subset_MJ, label='MJ')

sns.distplot(subset_LBJ, label='LBJ')

sns.distplot(subset_kobe, label='kobe')

plt.legend()
f, (ax,ax1,ax2) = plt.subplots(1,3, figsize=(12,8))

ax.boxplot(subset_MJ['PTS/MP'])

ax1.boxplot(subset_LBJ['PTS/MP'])

ax2.boxplot(subset_kobe['PTS/MP'])
def Kruskal_ANOVA(obj1, obj2, obj3):

    anova = stats.kruskal(obj1, obj2, obj3)

    return anova



Kruskal_ANOVA(subset_MJ,subset_LBJ,subset_kobe)