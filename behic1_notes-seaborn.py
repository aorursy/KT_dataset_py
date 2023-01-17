# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/Pokemon.csv")
data.head()
data.info()
data.notnull().all()
print( len(data[data["Type 2"].isnull()]) )
data[data["Type 2"].isnull()].head()
data["Type 1"].unique()
seri = data.groupby("Type 1").Attack.mean().sort_values(ascending = True)
seri
normalized_data = data[['#', 'Attack', 'Defense']].copy()
normalized_data.Attack = data.Attack / data.Attack.max()
normalized_data.Defense = data.Defense / data.Defense.max()
normalized_data.head(5)
plt.figure(figsize = (15,7))
sns.barplot(x=seri.keys(), y=seri.values)
plt.xticks(rotation=90)
plt.xlabel("Pokemon Type")
plt.ylabel("Attack")
plt.title("Attacks According To Pokemon Types")
plt.show()
plt.figure(figsize = (15, 7))
sns.pointplot(x='#', y="Attack", data=normalized_data.head(50), color='lime', alpha=0.7)
sns.pointplot(x='#', y="Defense", data=normalized_data.head(50), color='red', alpha=0.7)
plt.text(35, 0.6, 'Attack', color='lime', fontsize=17, style='italic')
plt.text(35, 0.55, 'Defense', color='red', fontsize=17, style='italic')
plt.xlabel("Pokemons", fontsize=14)
plt.ylabel("Values", fontsize=14)
plt.title("Attack&Defense Corrrelation", fontsize=15)
plt.show()
generations = data.Generation.value_counts()
plt.figure(figsize=[7,7])
#Explode ???
plt.pie(generations.values, explode = [0, 0, 0 ,0 ,0 ,0], labels = generations.index,  autopct = "%1.1f%%")
plt.title("Generation Types", size=15)
plt.show()

g = sns.jointplot(normalized_data.Attack, normalized_data.Defense,
              data=normalized_data.head(50), kind="kde", height=7)
plt.show()
sns.lmplot(x='Attack', y="Defense", data=normalized_data.head(50))
plt.show()
sns.kdeplot(normalized_data.head(50).Attack, normalized_data.head(50).Defense)
plt.show()
sns.kdeplot(normalized_data.head(50).Attack, normalized_data.head(50).Defense, shade=True)
plt.show()
sns.kdeplot(normalized_data.head(50).Attack, normalized_data.head(50).Defense, shade=True, cut=1)
plt.show()
plt.figure(figsize=(8,5))
pal = sns.cubehelix_palette(2, rot=.5, dark=.3)
sns.violinplot(data=normalized_data[["Attack", "Defense"]], palette=pal, inner="points")
plt.show()
plt.figure(figsize=(6,6))
sns.heatmap( normalized_data[["Attack", "Defense"]].corr(), annot=True,linewidth=.5, linecolor="red", fmt=".1f")
plt.show()

df = data[["Attack", "Generation", "Legendary"]]
df.head()
plt.figure(figsize=(16,6))
sns.boxplot(x="Generation",y="Attack", hue="Legendary", data=df, color="lime")
plt.show()
plt.figure(figsize=(16,6))
sns.swarmplot(x="Generation",y="Attack", hue="Legendary", data=df)
plt.show()
plt.figure(figsize=[12,6])
sns.countplot(data["Type 1"])
plt.xticks(rotation=90)
plt.show()