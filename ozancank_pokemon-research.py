# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns  # visualization tool



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data=pd.read_csv("../input/Pokemon.csv", index_col=0)
data.info()
a = data.head(10)

b = data.tail(10)

result=pd.concat([a,b])

display(result)
print(data['Type 1'].value_counts(dropna=False))
# try:

#    print("Choose generation (1-6): ")

#    selection=int(input())

#except ValueError:

#    selection=0



selection=3



if (selection>=1)and(selection<=6):

    display(data[(data.Legendary)&(data.Generation==selection)])

else:

    print("Invalid generation")
data[data.Attack==max(data.Attack)]
data[data.Defense==min(data.Defense)]
data.corr()
sns.lmplot(x="Attack",y="Defense", fit_reg=False, hue='Generation', data=data)

# x="Attack",y="Defense" : We write x and y axis tags. 

# fit_reg=False : We remove the regression line.

# hue='Generation': We color according to the generation.

# data=data : We determine the dataset.

plt.ylim(0, None) # We start 0 of y-axis value.

plt.xlim(0, None) # We start 0 of x-axis value.

plt.show() 
feature_data = data.drop(['Total', 'Generation', 'Legendary'], axis = 1) # We have removed 3 columns. Because it's not war values.

sns.boxplot(data = feature_data, notch = True, linewidth = 0.5, width = 0.6)

# notch: With this argument we show the limits of the media in the 95% confidence interval.

# linewidth: With this argument, we determine the line thickness of each box.

# width: With this argument we determine the width of each box.

plt.show()
f,ax=plt.subplots(figsize=(15,15)) #boyut ayarlıyoruz

pkmn_type_colors = ['#78C850',  # Grass

                    '#F08030',  # Fire

                    '#6890F0',  # Water

                    '#A8B820',  # Bug

                    '#A8A878',  # Normal

                    '#A040A0',  # Poison

                    '#F8D030',  # Electric

                    '#E0C068',  # Ground

                    '#EE99AC',  # Fairy

                    '#C03028',  # Fighting

                    '#F85888',  # Psychic

                    '#B8A038',  # Rock

                    '#705898',  # Ghost

                    '#98D8D8',  # Ice

                    '#7038F8',  # Dragon

                    '#390072',  # Dark

                    '#6a776c',  # Steel

                    '#5dc6c3',  # Flying

                   ] # We have set a special color according to the type of Pokemon.

sns.set_style("whitegrid")

sns.violinplot(x="Type 1",y="Attack",data=data,fmt='.1f',ax=ax, linewidth = 1.0, palette=pkmn_type_colors)



plt.show()
sns.set_style("whitegrid")

#sns.swarmplot(x="Type 1",y="Attack",data=data, palette=pkmn_type_colors, size=3)

sns.catplot(x="Type 1",y="Attack",data=data, kind="swarm", palette=pkmn_type_colors, height=6, aspect=4)

plt.show()
sns.violinplot(x = 'Type 1', y = 'Attack', data = data, inner = None, palette=pkmn_type_colors)

sns.swarmplot(x = 'Type 1', y = 'Attack', data = data, color = 'k', alpha = 0.7)

plt.title('Attack by Type')

plt.show()
melted_df = pd.melt(data, 

                    id_vars=["Name", "Type 1", "Type 2"], # Variables to keep

                    var_name="Stat") # Name of melted variable

melted_df.head()
print( data.shape )

print( melted_df.shape )
plt.figure(figsize=(10,6))

 

sns.swarmplot(x='Stat', y='value', data=melted_df, hue='Type 1', dodge=True, palette=pkmn_type_colors)

 

plt.ylim(0, 260)

 

plt.legend(bbox_to_anchor=(1, 1), loc=2)

plt.show()
corr = data.corr()



sns.heatmap(corr)



plt.show()
sns.distplot(data.Attack)

plt.show()
sns.countplot(x='Type 1', data=data, palette=pkmn_type_colors)

 

# X eksenindeki etiketleri döndür

plt.xticks(rotation=-45)

plt.show()
g = sns.factorplot(x = 'Type 1', y = 'Attack', data = data, hue = 'Generation', col = 'Generation', kind = 'swarm')

g.set_xticklabels(rotation = -45)

plt.show()
sns.kdeplot(data.Attack, data.Defense)

plt.show()
sns.jointplot(x = 'Attack', y = 'Defense', data = data)

plt.show()