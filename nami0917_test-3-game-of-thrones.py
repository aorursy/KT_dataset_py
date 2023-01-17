# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib as mpl

import matplotlib.pyplot as plt 

import seaborn as sns



# Set parameters on plt

plt.rcParams["axes.labelsize"] = 16.

plt.rcParams["xtick.labelsize"] = 13.

plt.rcParams["ytick.labelsize"] = 13.

plt.rcParams["legend.fontsize"] = 11.

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
battles = pd.read_csv("../input/battles.csv")

c_deaths = pd.read_csv("../input/character-deaths.csv")

c_predictions = pd.read_csv("../input/character-predictions.csv")
battles.head(10)
battles.describe()
battles.corr()
battles['attacker_outcome'].value_counts().plot(kind='bar')
battles['region'].value_counts().plot(kind='bar')
p = battles.groupby('year').sum()[["major_death", "major_capture"]].plot(kind = 'bar', figsize = (15, 6), rot = 0)

_ = p.set_xlabel("Year"), p.set_ylabel("No. of Death/Capture Events"), p.legend(["Major Deaths", "Major Captures"])
#numdefenders = number of troops defedning in a battle

#numattackers = number of troops attacking in a battle 

battles.loc[:, "numdefenders"] = (4 - battles[["defender_1", "defender_2", "defender_3", "defender_4"]].isnull().sum(axis = 1))

battles.loc[:, "numattackers"] = (4 - battles[["attacker_1", "attacker_2", "attacker_3", "attacker_4"]].isnull().sum(axis = 1))



houses = battles.groupby('battle_number').sum()[["numdefenders", "numattackers"]].plot(kind = 'bar', figsize = (10,6), rot = 0)

_ = houses.set_xlabel("Battle Number"), houses.set_ylabel("Number of Houses Participated"), houses.legend(["Defenders", "Attackers"])
data1 = battles.dropna(axis = 0, subset = [["attacker_size", "defender_size", "attacker_outcome"]]).copy(deep = True)

col = [sns.color_palette()[1] if x == "win" else "lightgray" for x in data1.attacker_outcome.values]

plot1 = data1.plot(kind = "scatter", x = "attacker_size", y = "defender_size", c = col, figsize = (10, 6), s = 100, lw = 2.)

_ = plot1.set_xlabel("Attacking Size"), plot1.set_ylabel("Defending Size")
correlation=data1.corr()

plt.figure(figsize=(10,5))

sns.heatmap(correlation, vmax=1, square=True,annot=True,cmap='cubehelix')



plt.title('Correlation between troop size and battle outcome')