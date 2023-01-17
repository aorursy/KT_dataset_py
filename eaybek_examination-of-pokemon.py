# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
import matplotlib.pyplot as plt
# Any results you write to the current directory are saved as output.
import seaborn
from alyssum.seperators import Title
from alyssum.decorators import run_once_message
from alyssum.cleaners import Cleaner
from alyssum import initialize
initialize()
blue = Title()
green = Title(color="green")
green.configure(length=40,design_char="-")
red = Title(color="red")
red.configure(design_char="*")
yellow = Title(color="yellow")
yellow.configure(design_char="+")
blue.write("hello")
green.write("hello")
red.write("hello")
yellow.write("hello")
pokemon = pd.read_csv("../input/Pokemon.csv")
blue.write("pokemon head")
pokemon.head()
blue.write("pokemon tail")
pokemon.tail()
blue.write("pokemon describe")
pokemon.describe()
blue.write("correlation")
pokemon.corr()
plt.figure(figsize=(12,12))
seaborn.heatmap(pokemon.corr(),annot=True)
plt.show()
blue.write("POKEMON INFO")
pokemon.info()
print(Cleaner.clean("#"))
print(Cleaner.clean("1MY UG##ly          (      )##tiTlE9"))
print(Cleaner.clean("MY UG##ly          (      )##tiTlE9"))
print(Cleaner.clean("Hello World!"))
pokemon.head(0)
pokemon.columns=[Cleaner.clean(each) for each in pokemon.columns]
pokemon.head(0)
t1=dict(pokemon.type_1.value_counts())
t2=dict(pokemon.type_2.value_counts())
keyset=set() # a set is a special list which can contain only unique elements so 

for i,j in zip(t1,t2):
    keyset.add(i)
    keyset.add(j)
keyset
sum_of_types={}
for each in keyset:
    sum_of_types[each]=0
for each in keyset:
    if each in t1.keys():
        sum_of_types[each] += t1[each]
    if each in t2.keys():
        sum_of_types[each] += t2[each]
blue.write("Pokemon Niche")
plt.figure(figsize=(12,12))
grid = plt.GridSpec(3, 2, wspace=0.4, hspace=0.3)
plt.subplot(grid[0,0])
plt.title("type 1 niche")
plt.pie(pokemon.type_1.value_counts(), labels=pokemon.type_1.unique(),autopct='%1.1f%%')
plt.subplot(grid[0,1])
plt.title("type 2 niche")
plt.pie(pokemon.type_2.value_counts(), labels=pokemon.type_2.dropna().unique(),autopct='%1.1f%%')
plt.subplot(grid[1:,0:])
plt.title("sum niche")
plt.pie(sum_of_types.values(), labels=sum_of_types.keys(),autopct='%1.1f%%')

plt.show()
len(pokemon.name[pokemon.type_1==pokemon.type_2])
poke=pokemon.loc[:,["name","hp","attack","defense","sp_atk","sp_def","legendary"]]
poke["key"]=["1" for i in range(0,len(poke))]
poke_merge=pd.merge(poke,poke,on='key') # if you merge two tables with same key value you'll get cartesian product of table's like sql join
poke_merge.head()
yellow.configure(length=45).write("attack defense sp_atk sp_def boxplots")
plt.boxplot([pokemon.attack,pokemon.defense,pokemon.sp_atk,pokemon.sp_def])
plt.show()
a=poke_merge[poke_merge.attack_x>(poke_merge.defense_y+poke_merge.hp_y)]
poke_merge["a_nn"]=a.attack_x-(a.defense_y+a.hp_y) # Normal vs Normal
b=poke_merge[poke_merge.sp_atk_x>(poke_merge.sp_def_y+poke_merge.hp_y)]
poke_merge["b_ss"]=b.sp_atk_x-(b.sp_def_y+a.hp_y) # Special vs Special
c=poke_merge[poke_merge.sp_atk_x>(poke_merge.defense_y+poke_merge.hp_y)]
poke_merge["c_sn"]=c.sp_atk_x-(c.defense_y+a.hp_y) #Special vs Normal
d=poke_merge[poke_merge.attack_x>(poke_merge.sp_def_y+poke_merge.hp_y)]
poke_merge["d_ns"]=d.attack_x-(d.sp_def_y+a.hp_y) # Normal vs Special
poke_merge.head()
a_fights=poke_merge[pd.notna(poke_merge.a_nn)]
b_fights=poke_merge[pd.notna(poke_merge.b_ss)]
c_fights=poke_merge[pd.notna(poke_merge.c_sn)]
d_fights=poke_merge[pd.notna(poke_merge.d_ns)]
## This section can be run only 1 time
@run_once_message("it's already run if you sure to execution please run all sections")# this decorator need alyssum.initilize() be sure add top 
def scope():
    global a_fights, b_fights, c_fights, d_fights
    a_fights.columns = ['hit' if i == 'a_nn' else i for i in a_fights.columns]
    a_fights=a_fights.drop(["defense_x","sp_atk_x","sp_def_x","key","attack_y","sp_atk_y","sp_def_y","b_ss","c_sn","d_ns"],axis=1)
    b_fights.columns = ['hit' if i == 'b_ss' else i for i in b_fights.columns]
    b_fights=b_fights.drop(["defense_x","attack_x","sp_def_x","key","attack_y","sp_atk_y","defense_y","a_nn","c_sn","d_ns"],axis=1)
    c_fights.columns = ['hit' if i == 'c_sn' else i for i in c_fights.columns]
    c_fights=c_fights.drop(["defense_x","attack_x","sp_def_x","key","attack_y","sp_atk_y","sp_def_y","a_nn","b_ss","d_ns"],axis=1)
    d_fights.columns = ['hit' if i == 'd_ns' else i for i in d_fights.columns]
    d_fights=d_fights.drop(["defense_x","sp_atk_x","sp_def_x","key","attack_y","sp_atk_y","defense_y","a_nn","b_ss","c_sn"],axis=1)
scope()
def bar_plot(table):
    values=dict(table.name_x.value_counts())
    list(values.values())
    plt.figure(figsize=(24,6))
    plt.bar(list(values.keys())[0:700:15],tuple(values.values())[0:700:15])
    plt.xticks(rotation=90)
    plt.show()
a_fights.sort_values("hit",ascending=False).head()
bar_plot(a_fights)
b_fights.sort_values("hit",ascending=False).head()
bar_plot(b_fights)
c_fights.sort_values("hit",ascending=False).head()
bar_plot(c_fights)
d_fights.sort_values("hit",ascending=False).head()
bar_plot(d_fights)
blue.write("generation pokemon counts")
plt.figure(figsize=(12,12))
plt.pie(pokemon.generation.value_counts(), labels=pokemon.generation.unique(),autopct='%1.1f%%')
plt.show()
generation_types=[pokemon[pokemon.generation==each].type_1.value_counts() for each in pokemon.generation.unique()]
dict(generation_types[0])
hists={}

for i, generation in enumerate(generation_types):
    for key, value in dict(generation).items():
        if key not in hists:
            hists[key]=[0,0,0,0,0,0]
        hists[key][i]=value
    
        
hists
plt.figure(figsize=(24,18))
for i in range(0,6):
    plt.subplot(6,1,i+1)
    plt.subplots_adjust(bottom=-0.6)

    plt.title("pokemon type 1 values in generation {}".format(i+1))
    #plt.grid()
    plt.xticks(rotation=20)
    plt.bar(hists.keys(),[values[i] for key, values in hists.items()])
plt.show()

poke_merge.head()
poke_merge["h_d"]=poke_merge.hp_y + poke_merge.defense_y
poke_merge["h_sd"]=poke_merge.hp_y + poke_merge.sp_def_y
pool=poke_merge[(poke_merge.legendary_x==False)&(poke_merge.legendary_y == True )]

min_defense=pool.sort_values("h_d").head(1)
min_sp_def=pool.sort_values("h_sd").head(1)
min_defense
min_sp_def
threshold=min(
    int(min_defense.h_d),
    int(min_sp_def.h_sd)
)
threshold
len(pokemon[(pokemon.attack > threshold)|(pokemon.sp_atk > threshold)])