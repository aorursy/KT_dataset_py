import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



data = pd.read_csv("../input/pokemon.csv")

data.shape
data.head()
g1 = data.loc[data.generation==1,:]

# dataframe.plot.scatter() method

g1.plot.scatter('attack', 'defense'); # The ';' is to avoid showing a message before showing the plot
# plt.scatter() function

plt.scatter('attack', 'defense', data=g1);
# sns.scatterplot() function

sns.scatterplot('attack', 'defense', data=g1);
g1.plot.scatter('attack', 'defense', s = 40, c = 'orange', marker = 's', figsize=(8,5.5));
plt.figure(figsize=(10,7)) # Specify size of the chart

plt.scatter('attack', 'defense', data=data[data.is_legendary==1], marker = 'x', c = 'magenta')

plt.scatter('attack', 'defense', data=data[data.is_legendary==0], marker = 'o', c = 'blue')

plt.legend(('Yes', 'No'), title='Is legendary?')

plt.show()
plt.figure(figsize=(10,7))

sns.scatterplot(x = 'attack', y = 'defense', s = 70, hue ='is_legendary', data=data); # hue represents color
plt.figure(figsize=(10,7))

sns.scatterplot(x = 'attack', y = 'defense', s = 50, hue = 'is_legendary', 

                style ='is_legendary', data=data); # style represents marker
plt.figure(figsize=(11,7))

sns.scatterplot(x = 'attack', y = 'defense', s = 50, hue = 'type1', data=data)

plt.legend(bbox_to_anchor=(1.02, 1)) # move legend to outside of the chart

plt.title('Defense vs Attack for All Pokemons', fontsize=16)

plt.xlabel('Attack', fontsize=12)

plt.ylabel('Defense', fontsize=12)

plt.show()
water = data[data.type1 == 'water']

water.plot.scatter('height_m', 'weight_kg', figsize=(10,6))

plt.grid(True) # add gridlines

plt.show()
water.plot.scatter('height_m', 'weight_kg', figsize=(10,6))

plt.grid(True)

for index, row in water.nlargest(5, 'height_m').iterrows():

    plt.annotate(row['name'], # text to show

                 xy = (row['height_m'], row['weight_kg']), # the point to annotate 

                 xytext = (row['height_m']+0.2, row['weight_kg']), # where to show the text

                 fontsize=12)

plt.xlim(0, ) # x-axis has minimum 0

plt.ylim(0, ) # y-axis has minimum 0

plt.show()
plt.figure(figsize=(10,7))

sns.lmplot(x = "speed", y = "attack", data=data);
sns.lmplot(x = "speed", y = "attack", data=data, height = 7, aspect = 9/7); # Make a 9x7 size plot
sns.lmplot(x = "speed", y = "attack", hue = 'is_legendary', ci = None, data=data, height = 6, aspect = 8/6);