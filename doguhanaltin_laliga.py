# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from collections import Counter



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
laliga= pd.read_csv("../input/laliga_player_stats_english.csv")
laliga.info()
laliga.head()
laliga1=laliga[laliga["Games played"]>=10]
laliga["Passes"]=[  each*1000 if each<5 else each for each in laliga["Passes"] ]
team=list(laliga["Team"].unique())

passessum=[]

for each in team:

    x=laliga[laliga["Team"]==each]

    summ=sum(x["Passes"])

    passessum.append(summ)
team

di={"team":team,"teamsumpasses":passessum}

laligapassdf=pd.DataFrame(di)
newin=(laligapassdf["teamsumpasses"].sort_values(ascending=False)).index.values

sortedpass=laligapassdf.reindex(newin)
sns.barplot(x=sortedpass["team"],y=sortedpass["teamsumpasses"],)

plt.xlabel("Team")

plt.ylabel("Number of Passes")

plt.xticks(rotation=90,color="Red")

plt.title("Teams' Passes Number")

plt.grid()

plt.show()
yc=laliga["Yellow Cards"].sort_values(ascending=False).index.values

ycd=laliga.reindex(yc)

y=ycd.iloc[0:15,:]
plt.figure(figsize=(10,10))

sns.barplot(y=y["Name"],x=y["Yellow Cards"],alpha=0.7,color="Yellow",label="Yellow Card")

sns.barplot(y=y["Name"],x=y["Red Cards"],alpha=0.5,color="red",label="Red Card")

sns.barplot(y=y["Name"],x=y["Second Yellows"],alpha=0.8, color="purple",label="2.Yellow")

plt.xlabel("Card")

ply.ylabel("Player Name")

plt.title("")

plt.legend()

plt.xticks(rotation=90)

plt.show()
dn=y["Duels lost"]/max(y["Duels lost"])

yn=y["Yellow Cards"]/max(y["Yellow Cards"])

sns.pointplot(x=y["Name"],y=dn,color="Red")

sns.pointplot(x=y["Name"],y=yn)

plt.xticks(rotation=90)

sns.jointplot(x=laliga["Duels lost"], y=laliga["Yellow Cards"],kind="kde")
pos=list(laliga["Position"].unique())

poscard=[]

for a in pos:

    pc=laliga[laliga["Position"]==a]

    suma=sum(pc["Yellow Cards"])

    poscard.append(suma)



    

posdata=pd.DataFrame({"pos":pos,"cardn":poscard})




labels=posdata.pos

colors=["Red","Darkblue","Green","royalblue"]

sizes=posdata.cardn

explode=[0,0,0,0]

plt.figure(figsize=(8,7))

plt.pie(sizes,colors=colors,explode=explode,labels=labels,autopct="%1.1f%%")

plt.title("Yellow Cards Number According to Position")

plt.show()

sns.boxplot(x=laliga["Position"],y=laliga["Yellow Cards"])
sns.swarmplot(y=laliga["Yellow Cards"],x=laliga["Red Cards"],hue=laliga["Position"])

plt.legend()
sns.countplot(laliga["Team"])

plt.xticks(rotation=90)

plt.title("Team's Player Numbers")
sns.lmplot(x="Shots",y="Offsides",data=laliga)

plt.show()