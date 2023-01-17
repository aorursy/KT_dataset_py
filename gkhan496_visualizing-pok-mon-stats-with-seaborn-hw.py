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
data =pd.read_csv("../input/Pokemon.csv")
data.head()
data.info()
df = pd.DataFrame(data.Attack)

dfAttack = df.sort_values(by="Attack",ascending=False).index.values

sorted_data_by_attack = data.reindex(dfAttack)

sorted_data = pd.concat([sorted_data_by_attack.head(),sorted_data_by_attack.tail()],axis=0,ignore_index=True)





f,ax = plt.subplots(figsize = (9,15))

sns.barplot(x=sorted_data.Name,y=sorted_data.Attack)

ax.legend(loc='lower right',frameon = True)     # legendlarin gorunurlugu

plt.xlabel("Pokemon Name")

plt.ylabel("Attack Rate")

plt.xticks(rotation=60)

plt.show()
attack_data= data["Attack"]

defens_data = data["Defense"]

speed_data = data["Speed"]

hp_data = data["HP"]

dataT=[attack_data,defens_data,speed_data,hp_data]

for i in range(4):

    dataT[i]=dataT[i]/max(dataT[i])

sorted_data = pd.concat([dataT[2],dataT[0],dataT[1],dataT[3],data["Name"],data["Legendary"],data["Type 1"],data["Generation"]],axis=1)

sorted_data.sort_values('Speed',inplace=True)

data_i = sorted_data.head(40)

f,ax1 = plt.subplots(figsize =(30,10))

props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

colors=['blue','red','black','yellow']

dataP= [data_i["Attack"],data_i["Defense"],data_i["Speed"],data_i["HP"]]

dataN= ["Attack","Defense","Speed","HP"]

r=0.07

sns.set(font_scale=1.5)

for i in range(4):

    sns.pointplot(x=data_i["Name"],y=dataP[i],data=data,color=colors[i],alpha=0.8,linestyles="-.")

    ax1.text(38,0.7+r,dataN[i],color=colors[i],fontsize = 20,fontweight='ultralight',horizontalalignment='center',bbox=props)

    r+=0.07

plt.xticks(rotation=75)

plt.xlabel('Pokémon',fontsize = 25,color='blue')

plt.ylabel('Values',fontsize = 25,color='blue')

plt.grid(linestyle="--",drawstyle="steps")
g = sns.jointplot("Total", "Attack", data=data, kind="hex", size=7)

plt.show()
g = sns.lmplot("Speed", "Attack", data=sorted_data,hue="Legendary",palette="inferno",col="Legendary",size = 26,aspect=.5)

plt.show()
data["Type 1"].unique()
data["Type 1"].value_counts()
labels= data["Type 1"].value_counts().index

colors = ['green','red','blue','yellow','white','pink','gold','black','purple','plum','silver','violet','salmon','cyan','maroon','grey','brown','orange']

a = []

for i in colors:

    a.append(0)

sizes = data["Type 1"].value_counts().values

plt.figure(figsize = (20,20))

plt.pie(sizes, explode=a, labels=labels, colors=colors, autopct='%1.1f%%')

plt.savefig("pie.png")
data.corr()
#data.drop("#", inplace=True, axis=1)

f,ax = plt.subplots(figsize=(15, 15))

sns.heatmap(data.corr(), ax=ax,annot=True, linewidths=1,vmax=0.7)

plt.show()
f,ax = plt.subplots(figsize=(10, 10))

sns.violinplot(x="Generation", y="HP", data=sorted_data, inner=None)

sns.swarmplot(x="Generation",y="HP",hue="Legendary",data=sorted_data,color="yellow",size=8,ax=ax)

plt.savefig("swa-vio.png")

plt.show()
g = sns.PairGrid(sorted_data, vars=['Attack', 'Defense', 'Speed'],

                 hue='Legendary', palette='RdBu_r')

g.map(plt.scatter, alpha=0.8)

g.add_legend();

plt.savefig("Pair.png")