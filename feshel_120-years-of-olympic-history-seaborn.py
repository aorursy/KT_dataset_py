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
athlete = pd.read_csv("../input/athlete_events.csv") #Athlete Datas

regions = pd.read_csv("../input/noc_regions.csv")# Regions Datas
athlete.head()
regions.head()
athlete.info()
athlete['NOC'].unique()
new_athlete = athlete.dropna(subset=['Height']) #Cleaning NaN value

noc_list = list(new_athlete['NOC'].unique()) 

noc_ratio= [] #height ratios of regions

for i in noc_list:

    x = new_athlete[new_athlete['NOC']==i]

    heights = sum(x.Height)/len(x)

    noc_ratio.append(heights)

data = pd.DataFrame({'noc_list':noc_list,'noc_ratio':noc_ratio})   

new_index = (data['noc_ratio'].sort_values(ascending=False)).index.values

sorted_data = data.reindex(new_index)

sorted_data.sort_values(by=['noc_ratio'])





# visualization

plt.figure(figsize=(15,10))



sns.barplot(x=sorted_data['noc_list'][0:15],y=sorted_data['noc_ratio'][0:15])

plt.xticks(rotation=45)

plt.xlabel('NOC')

plt.ylabel('Heights')

plt.title('Heights give NOC')

plt.show()
# 3) regions who  won the most gold medals 15 gold medals //

medal_cleaning = athlete.dropna(subset=['Medal','NOC'])

medal_gold = medal_cleaning.Medal[medal_cleaning.Medal == 'Gold']

medal_noc = medal_cleaning.NOC[medal_cleaning.Medal == 'Gold']

data = pd.DataFrame({'Medal':medal_gold,'NOC':medal_noc})

sorting = data.NOC.value_counts()[:15]





#Visualization

plt.figure(figsize=(15,10))

ax = sns.barplot(x=sorting.index.values,y=sorting)



plt.xticks(rotation=45)

plt.xlabel('NOC',color='blue')

plt.ylabel('Gold Medal')

plt.title('15 regions who won the most gold medals')
cleaning_noc_medal = athlete.dropna(subset=['Medal','NOC'])



noc_list = list(cleaning_noc_medal['NOC'].unique())



medal_gold = []

medal_bronze = []

medal_silver = []

medal_nan = []



for i in noc_list:

    x = cleaning_noc_medal[cleaning_noc_medal['NOC'] == i]

    medal_gold.append(sum(x.Medal == 'Gold')/len(x))

    medal_silver.append(sum(x.Medal == 'Silver')/len(x))

    medal_bronze.append(sum(x.Medal == 'Bronze')/len(x))

    medal_nan.append(sum(x.Medal == '')/len(x))

    

#Visualization



f,ax = plt.subplots(figsize=(9,15))

sns.barplot(x=medal_gold,y=noc_list,color='green',alpha=0.5,label='Gold Medal')

sns.barplot(x=medal_silver,y = noc_list,color='blue',alpha=0.4,label='Silver Medal' )

sns.barplot(x=medal_bronze,y=noc_list,color='yellow',alpha=0.7,label='Bronze Medal')

sns.barplot(x=medal_nan,y=noc_list,color='red',alpha=0.5,label='No Medal')    



ax.legend(loc='upper right',frameon = True)

ax.set(xlabel='NOC',ylabel='MEDAL',title='medal rates of regions')
point_athlete = athlete.dropna(subset=['Height','Weight'])



point_noc_list = list(point_athlete['NOC'].unique())

point_height_ratio = []

point_weight_ratio = []



for i in point_noc_list:

    x = point_athlete[point_athlete['NOC']==i]

    heights = sum(x.Height)/len(x)

    point_height_ratio.append(heights)

    

    weight = sum(x.Weight)/len(x)

    point_weight_ratio.append(weight)

    

point_data = pd.DataFrame({'noc_list':point_noc_list,'height':point_height_ratio,'weight':point_weight_ratio})   

point_data = point_data.sort_values(by=['height'],ascending=False)





#Visualization



f,ax1 = plt.subplots(figsize=(20,10))

sns.pointplot(x='noc_list',y= 'weight' ,data=point_data[:50],color='lime',alpha=0.8)

sns.pointplot(x='noc_list',y='height' ,data=point_data[:50],color='blue',alpha=0.7)

plt.text(40,0.6,'Height Ratio',color='red',fontsize=17,fontstyle='italic')

plt.text(40,0.5,'Weight Ratio ',color='gray',fontsize=17,fontstyle='italic')

plt.xlabel('NOC',fontsize=15,color='blue')

plt.ylabel('Values',fontsize=15,color='blue')

plt.title('Height Ratio vs Weight Ratio',fontsize=20,color='blue')

plt.xticks(rotation=45)

plt.grid()

# noc = region

# point_data = used with point plot data 

# point_data.height = noc height ratio 

# point_data.weight = noc weight ratio 

g = sns.jointplot(point_data.height, point_data.weight, kind="kde", size=7)

plt.savefig('graph.png')

plt.show()
# you can change parameters of joint plot

# kind : { “scatter” | “reg” | “resid” | “kde” | “hex” }

# Different usage of parameters but same plot with previous one

g = sns.jointplot("height", "weight", data=point_data,size=5, ratio=3, color="r")
#

pie_data = sorting.head(7)

labels=pie_data.index

colors = ['grey','blue','red','green','brown','yellow','lime']

explode=[0.03,0.02,0,0,0,0,0]

sizes = pie_data.values



#Visualization



plt.figure(figsize=(7,7))

plt.pie(sizes,explode=explode,labels=labels,colors=colors,autopct='%1.1f%%')

plt.title('Gold Medal Ratios',color='blue',fontsize=15)

plt.show()
#Visualization

sns.lmplot(x="height", y="weight", data=point_data)

plt.show()
kde_data = point_data



#Visualization

sns.kdeplot(kde_data.height,kde_data.weight,shade=True,cut=3)

plt.show()
#Violin Plot 

violin_data = point_data



#Visualization

pal = sns.cubehelix_palette(2,rot=-.5,dark=.3)

sns.violinplot(data=violin_data,palette=pal,inner='points')

plt.show()

#Heat Map

map_data = point_data

map_data.corr()
#correlation map 

#visualiztion height ratio vs weight ratio

f,ax = plt.subplots(figsize=(5, 5))

sns.heatmap(map_data.corr(), annot=True, linewidths=0.5,linecolor="red", fmt= '.1f',ax=ax)

plt.show()
box_data.head()


box_data = athlete.dropna(subset=['Medal']) #Cleanin data 



#Visualization

sns.boxplot(x='Sex',y='Age',hue='Medal',data=box_data,palette='PRGn')

plt.show()
#swarm_data.head()
#Swarm Plot

swarm_data = box_data[:1000]

#swarm_data = box_data



#Visualization

sns.swarmplot(x='Sex',y='Age',hue='Medal',data = swarm_data)

plt.show()


medal_cleaning = athlete.dropna(subset=['Medal','NOC'])

medal_cleaning_medal = medal_cleaning['Medal']

medal_cleaning_noc = medal_cleaning['NOC']

count_data = pd.DataFrame({'Medal':medal_cleaning_medal,'NOC':medal_cleaning_noc})



#Visualization

plt.figure(figsize=(10,7))

sns.countplot(data=count_data,x='Medal')

plt.title("Medal Count",color='blue',fontsize=13)