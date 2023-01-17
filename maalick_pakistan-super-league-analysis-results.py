# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
t20_data = pd.read_csv('../input/t20_matches.csv',usecols=[5,6,8,9,10,14,15])

t20_data.columns
t20_UAE = t20_data.query('venue == ["Abu Dhabi","Sharjah","Dubai (DSC)"] ').reset_index()

t20_psl = t20_UAE.query('home == ["Quetta Gladiators","Karachi Kings","Peshawar Zalmi","Islamabad United","Lahore Qalandars"] ').reset_index()

sns.set(style="darkgrid")

ax = sns.countplot(x= "winner", data=t20_psl, palette = "Greens_d")

#ax.set_xticklabels(rotation=30)

#locs, labels = plt.xticks()

#plt.setp(labels, rotation=20)

t20_psl['year'] = pd.DatetimeIndex(t20_psl['date']).year

sns.set(style="darkgrid")

ax = sns.countplot(x= "winner",hue = "year", data=t20_psl, palette = "RdBu")

locs, labels = plt.xticks()

plt.setp(labels, rotation=20)
win = t20_psl[t20_psl.winner == t20_psl.innings1]

no_of_wins = len(win.index)

total = len(t20_psl.index)

no_of_loss = total-no_of_wins

sizes = [(no_of_wins/total)*100, (no_of_loss/total)*100]

labels = ["Wins", "Loss"]

colors = ['salmon', 'olive']

patches, texts = plt.pie(sizes, colors=colors, explode=(0, 0.15),

         shadow=True, startangle=90)

plt.legend(patches, labels, loc="best")

plt.title("Win percentage batting first")

#plt.show()
runs = t20_psl[t20_psl.innings1_runs > 150]

total = len(runs.index)

wins = runs[runs.winner == runs.innings1]

win = len(wins.index)

loss = total - win

sizes = [(win/total)*100, (loss/total)*100]

labels = ["Wins", "Loss"]

colors = ['salmon', 'olive']

patches, texts = plt.pie(sizes, colors=colors, explode=(0, 0.15),

         shadow=True, startangle=90)

plt.legend(patches, labels, loc="best")

plt.title("Win percentage after scoring more than 150")

plt.show()