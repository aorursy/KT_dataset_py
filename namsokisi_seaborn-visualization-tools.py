# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#read data
pokemon = pd.read_csv("../input/pokemon.csv")
pokemon.head()
pokemon.info()
#Bar Plot
plt.figure(figsize = (10,10))
sns.barplot(x = "Generation",y = "HP",data = pokemon)
plt.xticks(rotation = 45)
plt.yticks(rotation = 45)
plt.xlabel("Generations")
plt.ylabel("HP")
plt.title("Generations per HP")
plt.show()
pokemon.head()
#Point Plot
f, ax1 = plt.subplots(figsize = (20,15))
sns.pointplot(x = "Type 1",y = "Attack", data = pokemon,color = "blue")
sns.pointplot(x = "Type 1",y = "Attack", data = pokemon,color = "red")
plt.grid()
plt.xticks(rotation = 45)
plt.yticks(rotation = 45)
plt.xlabel("Type 1",color = "grey")
plt.ylabel("Attack",color = "pink")
#Joint Plot
jp = sns.jointplot(pokemon.Attack,pokemon.Generation)
plt.show()
#LM Plot
sns.lmplot(x = "Attack",y = "Defense",data = pokemon )
plt.xlabel("Attack",color = "r")
plt.xticks(rotation = 60)
plt.ylabel("Defense",color = "cyan")
plt.yticks(rotation = 60)
plt.title("Attack VS Defense",color ="yellow")
plt.show()
#KDE Plot
sns.kdeplot(pokemon.Attack,pokemon.Defense)
#Violin Plot
palle = sns.cubehelix_palette(1, rot=-.6, dark=.4)
sns.violinplot(data=pokemon,palette=palle, inner ="points")
plt.xticks(rotation =15)
pokemon.corr()
#Heatmap
f, ax =plt.subplots(figsize = (10,10))
sns.heatmap(pokemon.corr(),linewidths=0.4,linecolor="black",alpha=0.8)
plt.xticks(rotation=90,color="blue")
plt.yticks(color="blue")
#Box Plot
sns.boxplot(x = "Generation", y = "HP",data = pokemon)
#Swarm Plot
sns.swarmplot(x = "Generation",y = "HP", data = pokemon)
#Pair Plot
sns.pairplot(pokemon)
plt.show()




