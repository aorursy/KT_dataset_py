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
data=pd.read_csv('../input/database.csv')
data.tail()
data.rename(columns={'Crime Type':'Crime_Type','Victim Sex':'V_Sex','Victim Age':'V_Age','Victim Race':'V_Race','Perpetrator Race':'P_race','Perpetrator Age':'P_Age','Perpetrator Sex':'P_Sex','Victim Count':'V_Count','Perpetrator Count':'P_Count'},inplace=True)
data.head()
state_list=list(data['State'].unique())

victim=[]

perpetrator=[] 





for i in state_list:

    x=data[data.State==i]

    victim.append(sum(x.V_Count))

    perpetrator.append(sum(x.P_Count))

    

datanew=pd.DataFrame({'state_list':state_list,'victim_nu':victim,'perpetrator':perpetrator})

new_index = (datanew['victim_nu'].sort_values(ascending=False)).index.values

sorted_data = datanew.reindex(new_index)



# visualization

plt.figure(figsize=(15,10))

sns.barplot(x=sorted_data['state_list'], y=sorted_data['victim_nu'])

plt.xticks(rotation= 90)

plt.xlabel('States')

plt.ylabel('Victim Numbers')

plt.title('Victim Numbers Given States')
datanew.info()
data.V_Race.dropna(inplace = True)

labels = data.V_Race.value_counts().index

colors = ['grey','blue','red','yellow','green','brown']

explode = [0,0,0,0,0,0]

sizes = data.V_Race.value_counts().values





plt.figure(figsize = (7,7))

plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')

plt.title('Victims According to Races',color = 'blue',fontsize = 15)
data.head()
data.Weapon.dropna(inplace = True)

labels = data.Weapon.value_counts().index

colors = ['grey','blue','red','yellow','green','brown']

explode = [0,0,0,0,0,0]

sizes = data.Weapon.value_counts().values





plt.figure(figsize = (7,7))

plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')

plt.title('Weapons types',color = 'blue',fontsize = 15)
data.head()
state_list=list(data['State'].unique())

v_race_commun=[]

p_race_commun=[]

for i in state_list:

    x=data[data['State']==i]

    r_count = Counter(x.V_Race)

    most_com = r_count.most_common(1)

    v_race_commun.append(most_com)

    p_count = Counter(x.P_race)

    cost_com = p_count.most_common(1)

    p_race_commun.append(cost_com)

    

datanew['most_vic_race']=v_race_commun

datanew['most_per_race']=p_race_commun
datanew.head(25)
g = sns.jointplot(datanew.victim_nu, datanew.perpetrator, kind="kde", size=10)

plt.savefig('graph.png')

plt.show()
datanew.head()
sns.jointplot("victim_nu", "perpetrator",data=datanew, kind="hex", color="#4CB391")
g = sns.jointplot("victim_nu", "perpetrator", data=datanew,size=5, ratio=3, color="r")
sns.kdeplot(datanew.victim_nu, datanew.perpetrator, shade=True, cut=3)

plt.show()
data.info()
sns.set_style('whitegrid')



sns.countplot(x='V_Sex', data= data)

sns.set_style('whitegrid')



sns.countplot(x='P_Sex', data= data)
data.head()
sns.countplot(x='Crime_Type', hue='P_Sex', data= data,palette='RdBu_r')
f,ax = plt.subplots(figsize=(5, 5))

sns.heatmap(data.corr(), annot=True, linewidths=0.5,linecolor="red", fmt= '.1f',ax=ax)

plt.show()
data.P_Age.value_counts()
data["P_Age"] = pd.to_numeric(data["P_Age"], errors='coerce')
data.info()
datanew.head()
sns.lmplot(x="victim_nu", y="perpetrator", data=datanew)

plt.show()
sns.boxplot(x="V_Sex", y="V_Age", hue="Crime_Type", data=data, palette="PRGn")

plt.show()
data.head()
g=sns.relplot(x="State", y="Year", hue="V_Sex", size="V_Age",

            sizes=(0, 100), alpha=.5, palette="muted",

            height=6, data=data)

g.set_xticklabels(rotation=90)