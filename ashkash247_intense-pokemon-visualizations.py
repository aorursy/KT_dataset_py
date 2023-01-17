# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)

import plotly.graph_objs as go


# Any results you write to the current directory are saved as output.
df_main = pd.read_csv("../input/Pokemon.csv")
df_type_mean = df_main.iloc[:,[2,4,5,6,7,8,9,10]]
df_type_mean= df_type_mean.groupby('Type 1',as_index=False).mean()

plt.style.use('fivethirtyeight')
plt.figure(figsize=(15,15))
plt.subplot(2,1,1)
g=sns.lineplot(x="Type 1",y="HP",data= df_type_mean)

sns.lineplot(x="Type 1",y="Attack",data= df_type_mean,color="green")
sns.lineplot(x="Type 1",y="Defense",data= df_type_mean,color="red")



blue_patch = mpatches.Patch(color='blue', label='HP')
green_patch = mpatches.Patch(color='green', label='Attack')
red_patch = mpatches.Patch(color='red', label='Defense')






g.set(xlabel='Type of pokemon', ylabel='Mean values of attributes')

plt.legend(handles=[blue_patch,green_patch,red_patch])


plt.subplot(2,1,2)
k= sns.lineplot(x="Type 1",y="Sp. Atk",data= df_type_mean,color="yellow")
sns.lineplot(x="Type 1",y="Sp. Def",data= df_type_mean,color="pink")
sns.lineplot(x="Type 1",y="Speed",data= df_type_mean,color="black")

yellow_patch = mpatches.Patch(color='yellow', label='Special Attack')
pink_patch = mpatches.Patch(color='pink', label='Special Defense')
black_patch = mpatches.Patch(color='black', label='Speed')


k.set(xlabel='Type of pokemon', ylabel='Mean values of attributes')
plt.legend(handles=[yellow_patch,pink_patch,black_patch])

df_legendary = df_main[df_main['Legendary']==True]
df_legen_count = df_legendary.groupby('Type 1').size().reset_index(name='counts')
#df_legen_count
plt.figure(figsize=(15,10))

#g=sns.barplot(x="Type 1",y="counts",data= df_legen_count,color='cyan').set_title('Count of legendary pokemons in each type')
countlegen = [go.Bar(
          x=df_legen_count['Type 1'] ,
    y=df_legen_count['counts'],
      marker=dict(
      color='rgb(49,130,189)'
    )
    )]
layout1 = go.Layout(
    title='Count of legendary pokemons in each type'
)
fig1 = go.Figure(data=countlegen, layout=layout1)
iplot(fig1)
# aggragation of water,normal and grass
df_normal = df_main[df_main['Type 1']=='Normal']
df_normal_mean = df_normal.mean().reset_index(name='mean_Values')
df_normal_mean=df_normal_mean.iloc[2:8,:]
df_normal_mean['Type']='Normal' 


df_water = df_main[df_main['Type 1']=='Water']
df_water_mean = df_water.mean().reset_index(name='mean_Values')
df_water_mean=df_water_mean.iloc[2:8,:]
df_water_mean['Type']='Water'

df_grass = df_main[df_main['Type 1']=='Grass']
df_grass_mean = df_grass.mean().reset_index(name='mean_Values')
df_grass_mean=df_grass_mean.iloc[2:8,:]
df_grass_mean['Type']='Grass'



df_appended_WNG = df_water_mean.append(df_normal_mean)

df_appended_WNG= df_appended_WNG.append(df_grass_mean)
#df_appended_WNG


#Aggregation of Bug,Psychic and Fire
df_bug = df_main[df_main['Type 1']=='Bug']
df_bug_mean = df_bug.mean().reset_index(name='mean_Values')
df_bug_mean=df_bug_mean.iloc[2:8,:]
df_bug_mean['Type']='Bug' 


df_psychic = df_main[df_main['Type 1']=='Psychic']
df_psychic_mean = df_psychic.mean().reset_index(name='mean_Values')
df_psychic_mean=df_psychic_mean.iloc[2:8,:]
df_psychic_mean['Type']='Psychic'

df_fire = df_main[df_main['Type 1']=='Fire']
df_fire_mean = df_fire.mean().reset_index(name='mean_Values')
df_fire_mean=df_fire_mean.iloc[2:8,:]
df_fire_mean['Type']='Fire'



df_appended_BPF = df_bug_mean.append(df_psychic_mean)

df_appended_BPF= df_appended_BPF.append(df_fire_mean)
#df_appended_BPF
# Aggregation of  Rock,Electric and Ground
df_rock = df_main[df_main['Type 1']=='Rock']
df_rock_mean = df_rock.mean().reset_index(name='mean_Values')
df_rock_mean=df_rock_mean.iloc[2:8,:]
df_rock_mean['Type']='Rock' 


df_electric = df_main[df_main['Type 1']=='Electric']
df_electric_mean = df_electric.mean().reset_index(name='mean_Values')
df_electric_mean=df_electric_mean.iloc[2:8,:]
df_electric_mean['Type']='Electric'

df_ground = df_main[df_main['Type 1']=='Ground']
df_ground_mean = df_ground.mean().reset_index(name='mean_Values')
df_ground_mean=df_ground_mean.iloc[2:8,:]
df_ground_mean['Type']='Ground'



df_appended_REG = df_rock_mean.append(df_electric_mean)

df_appended_REG= df_appended_REG.append(df_ground_mean)
#df_appended_REG
# Aggregation of Ghost,Dragon and dark
df_ghost = df_main[df_main['Type 1']=='Ghost']
df_ghost_mean = df_ghost.mean().reset_index(name='mean_Values')
df_ghost_mean=df_ghost_mean.iloc[2:8,:]
df_ghost_mean['Type']='Ghost' 


df_dragon = df_main[df_main['Type 1']=='Dragon']
df_dragon_mean = df_dragon.mean().reset_index(name='mean_Values')
df_dragon_mean=df_dragon_mean.iloc[2:8,:]
df_dragon_mean['Type']='Dragon'

df_dark = df_main[df_main['Type 1']=='Dark']
df_dark_mean = df_dark.mean().reset_index(name='mean_Values')
df_dark_mean=df_dark_mean.iloc[2:8,:]
df_dark_mean['Type']='Dark'



df_appended_GDD = df_ghost_mean.append(df_dragon_mean)

df_appended_GDD= df_appended_GDD.append(df_dark_mean)
#df_appended_GDD
# Aggregation of Poison,fighting and steel
df_poison = df_main[df_main['Type 1']=='Poison']
df_poison_mean = df_poison.mean().reset_index(name='mean_Values')
df_poison_mean=df_poison_mean.iloc[2:8,:]
df_poison_mean['Type']='Poison' 


df_fighting = df_main[df_main['Type 1']=='Fighting']
df_fighting_mean = df_fighting.mean().reset_index(name='mean_Values')
df_fighting_mean=df_fighting_mean.iloc[2:8,:]
df_fighting_mean['Type']='Fighting'

df_steel = df_main[df_main['Type 1']=='Ground']
df_steel_mean = df_steel.mean().reset_index(name='mean_Values')
df_steel_mean=df_steel_mean.iloc[2:8,:]
df_steel_mean['Type']='Steel'



df_appended_PFS = df_poison_mean.append(df_fighting_mean)

df_appended_PFS= df_appended_PFS.append(df_steel_mean)
#df_appended_PFS
# Aggregation of Ice,Fairy and flying
df_ice = df_main[df_main['Type 1']=='Ice']
df_ice_mean = df_ice.mean().reset_index(name='mean_Values')
df_ice_mean=df_ice_mean.iloc[2:8,:]
df_ice_mean['Type']='Ice' 


df_fairy = df_main[df_main['Type 1']=='Fairy']
df_fairy_mean = df_fairy.mean().reset_index(name='mean_Values')
df_fairy_mean=df_fairy_mean.iloc[2:8,:]
df_fairy_mean['Type']='Fairy'

df_flying = df_main[df_main['Type 1']=='Flying']
df_flying_mean = df_flying.mean().reset_index(name='mean_Values')
df_flying_mean=df_flying_mean.iloc[2:8,:]
df_flying_mean['Type']='Flying'



df_appended_IFF = df_ice_mean.append(df_fairy_mean)

df_appended_IFF= df_appended_IFF.append(df_flying_mean)
#df_appended_IFF
plt.style.use('bmh')


plt.figure(figsize=(16,20))

ax1=plt.subplot(3,2,1)
sns.barplot(x="index",y="mean_Values",hue='Type',data=df_appended_WNG,palette=["dodgerblue", "gold","green"]).set_title("Comparision between water,normal and grass")
ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1),ncol=3, fancybox=True, shadow=True)
ax1.set_ylim(0,90)


ax2=plt.subplot(3,2,2)
sns.barplot(x="index",y="mean_Values",hue='Type',data=df_appended_BPF,palette=["green", "pink","red"]).set_title("Comparision between Bug Phychic Fire")
ax2.legend(loc='upper center', bbox_to_anchor=(0.5, 1.02),ncol=3, fancybox=True, shadow=True)
ax2.set_ylim(0,110)

ax3=plt.subplot(3,2,3)
sns.barplot(x="index",y="mean_Values",hue='Type',data=df_appended_REG,palette=["brown", "yellow","pink"]).set_title("Comparision between rock electric ground")
ax3.legend(loc='upper center', bbox_to_anchor=(0.5, 1.03),ncol=3, fancybox=True, shadow=True)
ax3.set_ylim(0,110)


ax4=plt.subplot(3,2,4)
sns.barplot(x="index",y="mean_Values",hue='Type',data=df_appended_GDD,palette=["grey", "red","black"]).set_title("Comparision between Ghost dragon dark")
ax4.legend(loc='upper center', bbox_to_anchor=(0.65, 1),ncol=3, fancybox=True, shadow=True)
ax4.set_ylim(0,110)


ax5=plt.subplot(3,2,5)
sns.barplot(x="index",y="mean_Values",hue='Type',data=df_appended_PFS,palette=["forestgreen", "orange","grey"]).set_title("Comparision between poison fighting steel")
ax5.legend(loc='upper center', bbox_to_anchor=(0.5, 1),ncol=3, fancybox=True, shadow=True)
ax5.set_ylim(0,110)



ax6=plt.subplot(3,2,6)
sns.barplot(x="index",y="mean_Values",hue='Type',data=df_appended_IFF,palette=["cyan", "silver","indigo"]).set_title("Comparision between Ice Fairy Flying")
ax6.legend(loc='upper center', bbox_to_anchor=(0.5, 1),ncol=3, fancybox=True, shadow=True)
ax6.set_ylim(0,110)


df_main_nonull = df_main.dropna()
#df_main_nonull.head()

df_main_nonull =  df_main_nonull.groupby(['Type 1','Type 2']).size().reset_index(name='counts')
#df_main_nonull.head()
plt.figure(figsize=(20,10))
ax23=sns.barplot(x="Type 1",y="counts",data=df_main_nonull,hue="Type 2")
ax23.legend_.remove()
ax23.legend(loc='upper center', bbox_to_anchor=(0.5, 1),ncol=3, fancybox=True, shadow=True)

df_Genecount = df_main['Generation'].value_counts().to_frame()
df_Genecount['Gene'] = df_Genecount.index
df_Genecount.columns = ['Counts','Generation']
countGene = [go.Bar(
          x=df_Genecount['Generation'] ,
    y=df_Genecount['Counts'],
      marker=dict(
      color='rgb(49,130,189)'
    )
    )]

fig2= go.Figure(data=countGene)
iplot(fig2)
