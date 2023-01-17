# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import matplotlib.patches as patches

import matplotlib.pyplot as plt

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
pokeTable = pd.read_csv('../input/Pokemon.csv')

pokeTable.head(10)
pokeTable = pokeTable.set_index('Name')



pokeTable.index = pokeTable.index.str.replace(".*(?=Mega)", "")



pokeTable.head(10)
pokeTable = pokeTable.drop(['#'], axis = 1)



pokeTable.head(10)
# collects all types of pokemon

type1 = [pokeTable['Type 1'].value_counts()[key] for key in pokeTable ['Type 1'].value_counts().index]

type2 = [pokeTable['Type 2'].value_counts()[key] for key in pokeTable ['Type 1'].value_counts().index]



cords = np.arange(len(pokeTable['Type 1'].value_counts().index))



# width distance

width = .45



#random color 1 & 2

clr1 = np.random.rand(4)

clr2 = np.random.rand(4)



#bar graph

hndl = [patches.Patch(color = clr1, label = 'Type 1'), patches.Patch(color = clr2, label = 'Type 2')]

plt.bar(cords, type1, width, color = clr1)

plt.bar(cords + width, type2, width, color = clr2)

plt.gca().set_xticklabels(pokeTable['Type 1'].value_counts().index)

plt.gca().set_xticks(cords + width)

plt.xticks(rotation = 90)

plt.legend(hndl = hndl)
# This boxplot represents the minimum, median, and maximum base stats of all the pokemon types

pokeTableBP = pokeTable.drop(['Generation','Total'], axis = 1)

sns.boxplot(data = pokeTableBP)

plt.ylim(0,300)

plt.show()
# The boxplot explains the stats distribution of type 1 pokemon based on their attack

plt.subplots(figsize = (15,5))

plt.title('Attack Stats of Type 1 Pokemons')

sns.boxplot(x = 'Type 1', y = 'Attack', data = pokeTableBP)

plt.ylim(0,200)

plt.show()
# The boxplot explains the stats distribution of type 2 pokemon based on their attack

plt.subplots(figsize = (15,5))

plt.title('Attack Stats of Type 2 Pokemons')

sns.boxplot(x = 'Type 2', y = 'Attack', data = pokeTableBP)

plt.ylim(0,200)

plt.show()
# The boxplot explains the stats distribution of type 1 pokemon based on their defense

plt.subplots(figsize = (15,5))

plt.title('Defense Stats of Type 1 Pokemons')

sns.boxplot(x = 'Type 1', y = 'Defense', data = pokeTableBP)

plt.ylim(0,250)

plt.show()
# The boxplot explains the stats distribution of type 2 pokemon based on their defense

plt.subplots(figsize = (15,5))

plt.title('Defense Stats of Type 2 Pokemons')

sns.boxplot(x = 'Type 2', y = 'Defense', data = pokeTableBP)

plt.ylim(0,250)

plt.show()
# The boxplot explains the stats distribution of type 1 pokemon based on their speed

plt.subplots(figsize = (15,5))

plt.title('Speed Stats of Type 1 Pokemons')

sns.boxplot(x = 'Type 1', y = 'Speed', data = pokeTableBP)

plt.ylim(0,200)

plt.show()
# The boxplot explains the stats distribution of type 2 pokemon based on their speed

plt.subplots(figsize = (15,5))

plt.title('Speed Stats of Type 2 Pokemons')

sns.boxplot(x = 'Type 2', y = 'Speed', data = pokeTableBP)

plt.ylim(0,200)

plt.show()
plt.subplots(figsize = (20,10))

plt.title('Strongest Generation of Pokemon')

sns.violinplot(x = "Generation", y = "Total", data = pokeTable) #Kinda like boxplot but in a different perspective

plt.show()