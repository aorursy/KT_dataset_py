# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

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
df = pd.read_csv("../input/chess/games.csv")

df
winner_dist = df.winner.value_counts().reset_index()

print(winner_dist)

fig, ax = plt.subplots(figsize=(10,6))

plt.title("white, black  and  draw")

sns.barplot(x = "index", y = "winner", data = winner_dist)

ax.set_xlabel('white vs  black')

ax.set_ylabel('General outcomes')

plt.show()
winners_dist = df.winner.value_counts().reset_index().rename(columns={'index':'Exit', 'winner':'Total games'})

winners_dist.index += 1

print(winners_dist)





labels = winners_dist['Exit']

sizes = winners_dist['Total games']

fig1, ax1 = plt.subplots(figsize=(10,10))

explode = (0.05,0,0)

ax1.pie(sizes, explode=explode,labels=labels, autopct='%1.1f%%', startangle=70)

plt.title("Game outcomes by percentage")

plt.show()
opening_dist = df.opening_name.value_counts().reset_index()[:10]

print(opening_dist)

fig, ax = plt.subplots(figsize=(10,6))

plt.title("")

sns.barplot(x = "opening_name", y = "index", data = opening_dist)

ax.set_xlabel('opening game')

ax.set_ylabel('total quantity')

plt.show()