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
data_2017 = pd.read_csv('../input/2017.csv')
data_2016 = pd.read_csv('../input/2016.csv')
data_2015= pd.read_csv('../input/2015.csv')
data_2017.info

data_2016.info
data_2017.columns
data_2017.rename(
    columns={
   'Happiness.Score' : 'Happiness_Score',
   'Happiness.Rank' : 'Happiness_Rank',
   'Whisker.high' : 'Whisker_high',
   'Whisker.low' : 'Whisker_low',
   'Economy..GDP.per.Capita.' : 'Economy_GDP_per_Capita',
   'Health..Life.Expectancy.' : 'Health_Life_Expectancy',
   'Trust..Government.Corruption.' : 'Trust_Government_Corruption',
   'Dystopia.Residual' : 'Dystopia_Residual'
   
  },
  inplace=True
)
data_2017.columns
data_2017['Country'].unique()
data_2017['Happiness_Score']
#Happiness Scores of Countries in Bar Plot
plt.figure(figsize=(50,40))
sns.barplot(x=data_2017['Country'], y=data_2017['Happiness_Score'])
plt.xticks(rotation= 90)
plt.xlabel('Contries')
plt.ylabel('Happiness Scores')
plt.title('World Happiness Report')
#Happiness rate of each country
#For this dataset we dont need sorting but for other datasets we should do:
data_2017.Happiness_Score.replace(['-'],0.0,inplace = True)
data_2017.Happiness_Score = data_2017.Happiness_Score.astype(float)
country_list = list(data_2017['Country'].unique())
country_ratio = []
for i in country_list:
    x = data_2017[data_2017['Country']==i]
    country_rate = sum(x.Happiness_Score)/len(x)
    country_ratio.append(country_rate)
data = pd.DataFrame({'country_list': country_list,'country_ratio':country_ratio})
new_index = (data['country_ratio'].sort_values(ascending=False)).index.values
sorted_data = data.reindex(new_index)

#visualition 
plt.figure(figsize=(50,45))
sns.barplot(x=sorted_data['country_list'], y=sorted_data['country_ratio'])
plt.xticks(rotation= 90)
plt.xlabel('Countries')
plt.ylabel('Happiness Rate')
plt.title('World Happiness Report')
Country_list = list(data_2017['Country'].unique())
data_2017.Happiness_Score = data_2017.Happiness_Score.astype(float)
Score_list = list(data_2017['Happiness_Score'].unique())
A_list=[i for i in Country_list if i.startswith("A")]
A_Score_list=[]
for i in Country_list:
    if i.startswith("A"):
        x = data_2017[data_2017['Country']==i]
        S =x.Happiness_Score
        A_Score_list.append(S)
   
data = pd.DataFrame({'A_list': A_list,'A_Score_list':A_Score_list})
#new_index = (data['A_Score_list'].sort_values(ascending=False)).index.values
#sorted_data = data.reindex(new_index)

#A_Score_list=[i for i in data_2017  if i.startswith("A")]
plt.figure(figsize=(15,10))
sns.barplot(x=data.A_list, y=data.A_Score_list)
plt.xticks(rotation= 45)
plt.xlabel('Countries')
plt.ylabel('Happiness Rate')
plt.title('World Happiness Report')

#A_list=[i for i in Country_listif i.startswith("A")]

#data_2017.replace(['-'],0.0,inplace = True)
#data_2017.replace(['(X)'],0.0,inplace = True)

data_2017.loc[:,['Economy_GDP_per_Capita', 'Family', 'Health_Life_Expectancy', 'Freedom']] = data_2017.loc[:,['Economy_GDP_per_Capita', 'Family', 'Health_Life_Expectancy', 'Freedom']].astype(float)
Country_list = list(data_2017['Country'].unique())
Economy_GDP_per_Capita = list(data_2017['Economy_GDP_per_Capita'].unique())
Family = list(data_2017['Family'].unique())
Health_Life_Expectancy = list(data_2017['Health_Life_Expectancy'].unique())
Freedom = list(data_2017['Freedom'].unique())

# visualization
f,ax = plt.subplots(figsize = (15,25))
sns.barplot(x=Economy_GDP_per_Capita,y=Country_list,color='green',alpha = 0.5,label='Economy_GDP_per_Capita' )
sns.barplot(x=Family,y=Country_list,color='blue',alpha = 0.7,label='Family')
sns.barplot(x=Health_Life_Expectancy,y=Country_list,color='cyan',alpha = 0.6,label='Health_Life_Expectancy')
sns.barplot(x=Freedom,y=Country_list,color='red',alpha = 0.6,label='Freedom')

ax.legend(loc='lower right',frameon = True)     # legendlarin gorunurlugu



# visualize
f,ax1 = plt.subplots(figsize =(40,30))
sns.pointplot(x=data_2017.Country,y=data_2017.Happiness_Score,color='lime',alpha=0.8)
sns.pointplot(x=data_2017.Country,y=data_2017.Economy_GDP_per_Capita,color='red',alpha=0.8)
plt.text(0,0.5,'Economy',color='red',fontsize = 30,style = 'italic')
plt.text(0,0,'Happiness Score',color='lime',fontsize = 30,style = 'italic')
plt.xlabel('Countries',fontsize = 50,color='blue')
plt.ylabel('Values',fontsize = 50,color='blue')
plt.xticks(rotation= 90)
plt.title('Economy  VS  Happiness Score',fontsize = 50,color='blue')
plt.grid()

# pearsonr= if it is 1, there is positive correlation and if it is, -1 there is negative correlation.
# If it is zero, there is no correlation between variables
# Show the joint distribution using kernel density estimation 
g = sns.jointplot(data_2017.Happiness_Score, data_2017.Economy_GDP_per_Capita, kind="kde", size=7)
plt.savefig('graph.png')
plt.show()
g = sns.jointplot(data_2017.Happiness_Score, data_2017.Economy_GDP_per_Capita, kind="kde", color="g",size=7)
g.plot_joint(plt.scatter, c="w", s=30, linewidth=1, marker="+")
g.ax_joint.collections[0].set_alpha(0)
plt.savefig('graph.png')
plt.show()

g = sns.jointplot(data_2017.Happiness_Score, data_2017.Economy_GDP_per_Capita, kind="hex", color="g",size=7)
#g.plot_joint(plt.scatter, c="w", s=30, linewidth=1, marker="+")
#g.ax_joint.collections[0].set_alpha(0)
plt.savefig('graph.png')
plt.show()

g = sns.jointplot(data_2017.Happiness_Score, data_2017.Economy_GDP_per_Capita,size=5, ratio=3, color="r")
data=data_2017.Country.head(5)
data


data.value_counts()
labels = data.value_counts().index
colors = ['grey','blue','red','yellow','green']
explode = [0,0,0,0,0]
sizes = data.value_counts().values

# visual
plt.figure(figsize = (7,7))
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%')
plt.title('Countries in Data Dataset',color = 'blue',fontsize = 15)

data=data_2017
# lmplot 
# Show the results of a linear regression within each dataset
sns.lmplot(x="Happiness_Score", y='Economy_GDP_per_Capita', data=data)
plt.show()
# cubehelix plot
sns.kdeplot(data.Happiness_Score, data.Economy_GDP_per_Capita, shade=True, cut=2)
plt.show()
data=[data_2017.Happiness_Score]

# Show each distribution with both violins and points
# Use cubehelix to get a custom sequential palette
#plt.figure(figsize = (40,40))
pal = sns.cubehelix_palette(2, rot=-.5, dark=.3)
sns.violinplot(data=data, palette=pal, inner="points")

plt.show()
data_2017.corr()
#correlation map

f,ax = plt.subplots(figsize=(5, 5))
sns.heatmap(data_2017.corr(), annot=True, linewidths=0.5,linecolor="red", fmt= '.1f',ax=ax)
plt.show()
data1=data_2017['Family']
data2 = data_2017['Happiness_Score']
data3=data_2017['Happiness_Rank']
data4=data_2017['Economy_GDP_per_Capita']
data = pd.concat([data1,data2,data3,data4],axis =1) # axis = 0 : adds dataframes in row


# pair plot
sns.pairplot(data)
plt.show()
above5 =['above5' if i >=5.0 else 'below5' for i in data_2017.Happiness_Score]
df = pd.DataFrame({'Happiness_Score':above5})
sns.countplot(x=df.Happiness_Score)
plt.title('Happiness Score',color = 'blue',fontsize=15)

