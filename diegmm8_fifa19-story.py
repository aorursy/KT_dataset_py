# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 

from matplotlib.ticker import PercentFormatter



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
players = pd.read_csv("../input/data.csv")
print(players.head(15))

print(players.shape)

print(players.info())
pd.DataFrame(players['Nationality'].value_counts(dropna=False)).head(15)
players.describe()
top_players = players["Age"].loc[players['Overall']>=80].value_counts(bins=3, dropna=False)

top_players
data = top_players

total = data.sum(axis=0)

data = data/total

data.plot(kind='bar',figsize=(15,6))

plt.title("Age proportion among top players",fontsize= 16)

plt.gca().yaxis.set_major_formatter(PercentFormatter(1))

plt.gca().xaxis.set_ticklabels(["26 - 33 years old","17 - 25 years old", "34 - 40 years old"])

plt.show()
top_players = players.loc[players['Overall']>=90]

top_players
players.loc[players['Overall']>=86].groupby('Club').mean().sort_values(['Overall'],ascending=False)