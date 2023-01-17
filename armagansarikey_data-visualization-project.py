# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #for visualization
import seaborn as sns  #for visualization
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#read from the csv files 
data_2015 = pd.read_csv('../input/2015.csv')
data_2016 = pd.read_csv('../input/2016.csv')
#first 10 rows from data
data_2015.head(10)
data_2015.info()
# correlation
data_2015.corr()
#correlation map
#As seen from below heat map, economy, health and family are directly correlated with happiness score.
f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(data_2015.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()
data_2015.columns
#data_2015["Happiness Rank"].value_counts()
data_2015['Region'].unique()
# This graph shows us distribution of Happiness Score.
sns.distplot(data_2015['Happiness Score'], bins=20, kde=False, color='r')
sns.set(style='darkgrid',palette='muted',font_scale=1)
sns.distplot(data_2015['Generosity'], bins=20, kde=True, color='b')
sns.set(style='whitegrid',palette='muted',font_scale=1)
# Happiness Score of each region
data_2015["Happiness Score"].replace(['-'],0.0,inplace = True)
data_2015["Happiness_Score"]=data_2015["Happiness Score"]

area_list = list(data_2015['Region'].unique())
area_happiness_ratio = []
for i in area_list:
    x = data_2015[data_2015["Region"]==i]
    area_happiness_rate = sum(x.Happiness_Score)/len(x)
    area_happiness_ratio.append(area_happiness_rate)
data = pd.DataFrame({'area_list': area_list,'area_happiness_ratio':area_happiness_ratio})
new_index = (data['area_happiness_ratio'].sort_values(ascending=False)).index.values
sorted_data = data.reindex(new_index)

# visualization
plt.figure(figsize=(15,10))
sns.barplot(x=sorted_data['area_list'], y=sorted_data['area_happiness_ratio'])
plt.xticks(rotation= 45) 
plt.xlabel('Regions')
plt.ylabel('Happiness Score')
plt.title('Happiness Score Given Regions')
plt.show()
# Economy Rate Given Regions
data_2015["Economy (GDP per Capita)"].replace(['-'],0.0,inplace = True)
data_2015["Economy"]=data_2015["Economy (GDP per Capita)"]

area_list = list(data_2015['Region'].unique())
area_economy = []
for i in area_list:
    x = data_2015[data_2015["Region"]==i]
    area_economy_rate = sum(x.Economy)/len(x)
    area_economy.append(area_economy_rate)
# sorting
data2 = pd.DataFrame({'area_list': area_list,'area_economy_ratio':area_economy})
new_index = (data2['area_economy_ratio'].sort_values(ascending=True)).index.values
sorted_data2 = data2.reindex(new_index)
# visualization
plt.figure(figsize=(15,10))
sns.barplot(x=sorted_data2['area_list'], y=sorted_data2['area_economy_ratio'])
plt.xticks(rotation= 90)
plt.xlabel('Regions')
plt.ylabel('Economy Rate')
plt.title("Economy Rate Given Regions")
plt.show()
# Percentage of Regions' Rate According to Components that affected of happiness score
data_2015.replace(['-'],0.0,inplace = True)
data_2015.replace(['(X)'],0.0,inplace = True)
data_2015["family"]=data_2015["Family"]
data_2015["health"]=data_2015["Health (Life Expectancy)"]
data_2015["freedom"]=data_2015["Freedom"]
data_2015["trust"]=data_2015["Trust (Government Corruption)"]
data_2015["generosity"]=data_2015["Generosity"]

area_list = list(data_2015['Region'].unique())
family = []
health = []
freedom = []
trust = []
generosity = []
for i in area_list:
    x = data_2015[data_2015["Region"]==i]
    family.append(sum(x.family)/len(x))
    health.append(sum(x.health) / len(x))
    freedom.append(sum(x.freedom) / len(x))
    trust.append(sum(x.trust) / len(x))
    generosity.append(sum(x.generosity) / len(x))

# visualization
f,ax = plt.subplots(figsize = (9,15))
sns.barplot(x=family,y=area_list,color='green',alpha = 0.5,label='Family' )
sns.barplot(x=health,y=area_list,color='blue',alpha = 0.7,label='Health')
sns.barplot(x=freedom,y=area_list,color='cyan',alpha = 0.6,label='Freedom')
sns.barplot(x=trust,y=area_list,color='yellow',alpha = 0.6,label='Trust')
sns.barplot(x=generosity,y=area_list,color='red',alpha = 0.6,label='Generosity')

ax.legend(loc='lower right',frameon = True)     # legend 
ax.set(xlabel='Percentage of Components that affected of happiness score', ylabel='Regions',title = "Percentage of Regions' Rate According to Components that affected of happiness score ")
plt.show()
# Health vs Happiness Score of each country
data_2015["health"]=data_2015["Health (Life Expectancy)"]
data_2015["Happiness_Score"]=data_2015["Happiness Score"]

area_list = list(data_2015['Country'].unique())
area_happiness_ratio = []
for i in area_list:
    x = data_2015[data_2015["Country"]==i]
    area_happiness_rate = sum(x.Happiness_Score)/len(x)
    area_happiness_ratio.append(area_happiness_rate)

data = pd.DataFrame({'area_list': area_list,'area_happiness_ratio':area_happiness_ratio})
new_index = (data['area_happiness_ratio'].sort_values(ascending=False)).index.values
sorted_data = data.reindex(new_index)

area_list = list(data_2015['Country'].unique())
area_health = []
for i in area_list:
    x = data_2015[data_2015["Country"]==i]
    area_health_rate = sum(x.health)/len(x)
    area_health.append(area_health_rate)

# sorting
data3 = pd.DataFrame({'area_list': area_list,'area_health_ratio':area_health})
new_index = (data3['area_health_ratio'].sort_values(ascending=True)).index.values
sorted_data3 = data3.reindex(new_index)

sorted_data['area_happiness_ratio'] = sorted_data['area_happiness_ratio']/max( sorted_data['area_happiness_ratio'])
sorted_data3['area_health_ratio'] = sorted_data3['area_health_ratio']/max( sorted_data3['area_health_ratio'])
data = pd.concat([sorted_data3,sorted_data['area_happiness_ratio']],axis=1)
data.sort_values('area_health_ratio',inplace=True)

# visualization
f,ax1 = plt.subplots(figsize =(25,10))
sns.pointplot(x='area_list',y='area_health_ratio',data=data,color='lime',alpha=0.8)
sns.pointplot(x='area_list',y='area_happiness_ratio',data=data,color='red',alpha=0.8)
plt.text(90,0.2,'happiness score ratio',color='red',fontsize = 17,style = 'italic')
plt.text(90,0.15,'health ratio',color='lime',fontsize = 18,style = 'italic')
plt.xticks(rotation= 90)
plt.xlabel('Countries',fontsize = 15,color='blue')
plt.ylabel('Values',fontsize = 15,color='blue')
plt.title('Health vs Happiness Score',fontsize = 20,color='blue')
plt.grid()
plt.tight_layout
plt.show()
sns.jointplot(data_2015['Economy (GDP per Capita)'],data_2015['Happiness Score'],data=data_2015,kind='hex', color='y')
# Visualization of Health vs Happiness Score of each country with different style of seaborn code
# joint kernel density
# pearsonr= if it is 1, there is positive correlation and if it is, -1 there is negative correlation.
# If it is zero, there is no correlation between variables
# Show the joint distribution using kernel density estimation 
g = sns.jointplot(data.area_health_ratio, data.area_happiness_ratio, kind="kde", size=7)
plt.savefig('graph.png') #we save the figure
plt.show()
data_2015["family"]=data_2015["Family"]
data_2015["Happiness_Score"]=data_2015["Happiness Score"]

area_list = list(data_2015['Country'].unique())
area_family = []
for i in area_list:
    x = data_2015[data_2015["Country"]==i]
    area_family_rate = sum(x.family)/len(x)
    area_family.append(area_family_rate)

# sorting
data4 = pd.DataFrame({'area_list': area_list,'area_family_ratio':area_health})
new_index = (data4['area_family_ratio'].sort_values(ascending=True)).index.values
sorted_data4 = data4.reindex(new_index)

sorted_data['area_happiness_ratio'] = sorted_data['area_happiness_ratio']/max( sorted_data['area_happiness_ratio'])
sorted_data4['area_family_ratio'] = sorted_data4['area_family_ratio']/max( sorted_data4['area_family_ratio'])
data = pd.concat([sorted_data4,sorted_data['area_happiness_ratio']],axis=1)
data.sort_values('area_family_ratio',inplace=True)

# visualization
g = sns.jointplot(data.area_family_ratio, data.area_happiness_ratio, kind="kde", size=10, color='magenta' )
plt.savefig('graph.png')
plt.tight_layout
plt.show()
# you can change parameters of joint plot
# Different usage of parameters but same plot with previous one
g = sns.jointplot("area_family_ratio", "area_happiness_ratio", data=data,size=7, ratio=3, color="g")
data_2015.Region.value_counts()
#Counts According to Regions

labels = data_2015.Region.value_counts().index
colors = ['grey','blue','red','yellow','green','brown','purple','pink','orange','lime']
explode = [0,0,0,0,0,0,0,0,0,0]
sizes = data_2015.Region.value_counts().values

# visualization
plt.figure(figsize = (7,7))
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%')
plt.title('Counts According to Regions',color = 'blue',fontsize = 15)
plt.show()
data.head()
# Visualization of family ratio vs happiness ratio of each region with different style of seaborn library plots
# lmplot 
# Show the results of a linear regression within each dataset
sns.lmplot(x="area_happiness_ratio", y="area_family_ratio", data=data)
plt.show()
# Visualization of family ratio vs happiness ratio of each region with different style of seaborn code
# cubehelix plot
# same data different plot type
sns.kdeplot(data.area_happiness_ratio, data.area_family_ratio, shade=True, cut=3)
plt.show()
# Show each distribution with both violins and points
# Use cubehelix to get a custom sequential palette
pal = sns.cubehelix_palette(3, rot=-0.7, dark=0.3)
sns.violinplot(data=data, palette=pal, inner="points")
plt.show()
data.corr()
#correlation map
# Visualization of area_family_ratio vs area_happiness_ratio of each region with different style of seaborn code
f,ax = plt.subplots(figsize=(6, 6))
sns.heatmap(data.corr(), annot=True, linewidths=1,linecolor="lime", fmt= '.1f',ax=ax)
plt.show()
data_2015.Region.unique()
f,ax = plt.subplots(figsize=(18, 18))
sns.boxplot(x='Region', y='Happiness Score', data=data_2015, palette='BuGn_d')
plt.xticks(rotation= 90)
plt.tight_layout()
plt.show()
# swarm plot

f,ax = plt.subplots(figsize=(10, 10))
sns.swarmplot(x="Region", y="Happiness Score", data=data_2015)
plt.xticks(rotation= 90)
plt.tight_layout()
plt.show()
data.head()
# pair plot
sns.pairplot(data)
plt.show()
data_2015.Region.value_counts()
data_2015.head()
sns.countplot(data_2015.Region)
plt.title("Regions",color = 'blue',fontsize=15)
plt.xticks(rotation= 60)
plt.tight_layout
plt.show()