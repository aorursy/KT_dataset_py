# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re # regex
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#load both datasets
fifa = pd.read_csv("/kaggle/input/fifa19/data.csv", index_col="Name")
world_cup = pd.read_csv("/kaggle/input/wc2018-playerdata/2018  2019 World Cup - Dives Dataset (public) - 2018 Mens Match Data.csv", 
                        header=1)
#validation
#Run the following on both datasets:
fifa.head()
fifa.index
fifa.columns
fifa.dtypes
fifa.shape
#Bit of cleaning and where is Neymar
world_cup.Player = world_cup.Player.str.replace(r'(?<=\b\w)\w*\s', '. ')
world_cup.Player = world_cup.Player.str.replace('Neymar', 'Neymar Jr')
#join
dive_count = world_cup.value_counts('Player').rename('dives').reset_index()
df = dive_count.reset_index().merge(fifa, left_on='Player', right_on='Name')

print(dive_count.shape)
print(df.shape)
#correlation
f, ax = plt.subplots(figsize=(11, 9))
corr = df.corr()
sns.heatmap(corr)
#low correlation based on fifa attributes, but good penalty kicker tend to be good divers...
#I was expecting dribbling to be higher on that list
corr.dives.sort_values(ascending=False)
# mid-fielders are the highest divers
f, ax = plt.subplots(figsize=(11, 9))
sns.barplot(x='Position', y='dives', data=df)
# Brazil, France and England at the top
f, ax = plt.subplots(figsize=(11, 9))
ax = sns.catplot(x='dives', y='Nationality', data=df)
# Few outliers
f, ax = plt.subplots(figsize=(11, 9))
ax = sns.boxplot(df.dives)
print(df.loc[:15,['dives', 'Player']])
