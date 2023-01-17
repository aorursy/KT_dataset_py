import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")


import os
print(os.listdir("../input"))


data = pd.read_csv("../input/battles.csv")
data.info()
data.head()
pd.melt(frame=data, id_vars="year",value_vars="name")
x = pd.melt(frame=data,id_vars="name",value_vars="battle_type").groupby(data.year)
x.get_group(298)

df = data.groupby([data.year,data.attacker_1])
df.size()
major = data[data.major_death==1.0]
attacker = major.attacker_king.unique()
defender = [i for i in major.defender_king.unique() if i is not np.nan]
[i for i in attacker if i in defender]
l = []
for i in zip(list(data.attacker_size),list(data.defender_size)):
    l.append(sum(i))
max(l)
plt.figure(figsize=(20,7))
sns.swarmplot(y=data.year,hue = data.attacker_outcome,x=data.attacker_1)
plt.xlabel("Attacker")
plt.ylabel("Year")
plt.show()
plt.figure(figsize=(20,7))
sns.swarmplot(y=data.year,hue = data.attacker_outcome,x=data.defender_1)
plt.xlabel("Defender")
plt.ylabel("Year")
plt.show()
d = defaultdict(list)
data298 = data[data.year==298]
data298 = data298.fillna(0)

for i in data298.battle_type.unique():
    battle_type = data298[data298.battle_type == i].battle_type
    def_size = data298[data298.battle_type == i].defender_size
    for k,v in zip(list(battle_type),list(def_size)):
        d[k].append(v)
        
sns.barplot([k for k,v in d.items()],[sum(v)/len(v) for k,v in d.items()])
plt.xlabel("Battle Type")
plt.ylabel("Mean Man Power")
plt.show()

dataMajor = data[data.major_death==1.0]
stark = dataMajor[data.attacker_1 == "Stark"]
lannister = dataMajor[data.attacker_1 == "Lannister"]

SL = pd.concat([stark,lannister],axis=0)

plt.figure(figsize=(10,5))
sns.countplot(SL.region)
plt.xlabel("Region")
plt.ylabel("Count")
plt.show()
dataAttacker = data[data.attacker_1 == "Stark"]
dataSL = dataAttacker[dataAttacker.defender_1 == "Lannister" ]
dataSL = dataSL.fillna(0)

sns.scatterplot(x=dataSL.attacker_1,y=[i+j for i,j in zip(dataSL.attacker_size,dataSL.defender_size)],hue = dataSL.attacker_outcome)
plt.xlabel("Attacker")
plt.ylabel("Man Power")
plt.show()