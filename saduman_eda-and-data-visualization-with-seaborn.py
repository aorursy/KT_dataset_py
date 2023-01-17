# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns #for data visualization
import matplotlib.pyplot as plt  #for data visualization

import warnings            
warnings.filterwarnings("ignore") 

from pandas.tools.plotting import parallel_coordinates

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#load data from csv file
data_2015=pd.read_csv('../input/2015.csv')
data_2016=pd.read_csv('../input/2016.csv')
data_2017=pd.read_csv('../input/2017.csv')
#columns name change
data_2015.columns=[each.split()[0] if(len(each.split())>2) else each.replace(" ","_") for each in data_2015.columns]
data_2016.columns=[each.split()[0] if(len(each.split())>2) else each.replace(" ","_") for each in data_2016.columns]
data_2017.columns=[each.replace("."," ") for each in data_2017.columns]
data_2017.columns=[each.split()[0] if(len(each.split())>2) else each.replace(" ","_") for each in data_2017.columns]


data_2015.head()
#getting an overview of our data
data_2015.info()
print("Are There Missing Data? :",data_2015.isnull().any().any())
print(data_2015.isnull().sum())
#data_2015["Happiness_Score"].value_counts()
#we found out how many hospital country in our data
print("\n\nRegion in Dataset:\n")
print("There are {} different values\n".format(len(data_2015.Region.unique())))
print(data_2015.Region.unique())
region_lists=list(data_2015['Region'].unique())
region_happiness_ratio=[]
for each in region_lists:
    region=data_2015[data_2015['Region']==each]
    region_happiness_rate=sum(region.Happiness_Score)/len(region)
    region_happiness_ratio.append(region_happiness_rate)
    
data=pd.DataFrame({'region':region_lists,'region_happiness_ratio':region_happiness_ratio})
new_index=(data['region_happiness_ratio'].sort_values(ascending=False)).index.values
sorted_data = data.reindex(new_index)

sorted_data
#Visualization
#Create a new figure and make the size (12,10)
plt.figure(figsize=(12,10))
sns.barplot(x=sorted_data['region'], y=sorted_data['region_happiness_ratio'],palette=sns.cubehelix_palette(len(sorted_data['region'])))
# Place the region names at a 90-degree angle.
plt.xticks(rotation= 90)
plt.xlabel('Region')
plt.ylabel('Region Happiness Ratio')
plt.title('Happiness rate for regions')
plt.show()
#meaningless data control
#data_2015["Economy"].value_counts()
region_lists=list(data_2015['Region'].unique())
region_economy_ratio=[]
for each in region_lists:
    region=data_2015[data_2015['Region']==each]
    region_economy_rate=sum(region.Economy)/len(region)
    region_economy_ratio.append(region_economy_rate)
    
data_economy=pd.DataFrame({'region':region_lists,'region_economy_ratio':region_economy_ratio})
new_index_economy=(data_economy['region_economy_ratio'].sort_values(ascending=True)).index.values
sorted_data_economy = data_economy.reindex(new_index_economy)
sorted_data_economy.head()
#Visualization
#Create a new figure and make the size (12,10)
f,ax1 = plt.subplots(figsize =(12,10))
sns.barplot(x=sorted_data_economy['region'], y=sorted_data_economy['region_economy_ratio'],palette="rocket", ax=ax1)
# Place the region names at a 90-degree angle.
plt.xticks(rotation= 90)
plt.xlabel('Region')
plt.ylabel('Region Economy Ratio')
plt.title('Economy rate for regions')
plt.show()
#Horizontal bar plot
region_lists=list(data_2015['Region'].unique())
share_economy=[]
share_family=[]
share_health=[]
share_freedom=[]
share_trust=[]
for each in region_lists:
    region=data_2015[data_2015['Region']==each]
    share_economy.append(sum(region.Economy)/len(region))
    share_family.append(sum(region.Family)/len(region))
    share_health.append(sum(region.Health)/len(region))
    share_freedom.append(sum(region.Freedom)/len(region))
    share_trust.append(sum(region.Trust)/len(region))
#Visualization
f,ax = plt.subplots(figsize = (9,5))
sns.set_color_codes("pastel")
sns.barplot(x=share_economy,y=region_lists,color='g',label="Economy")
sns.barplot(x=share_family,y=region_lists,color='b',label="Family")
sns.barplot(x=share_health,y=region_lists,color='c',label="Health")
sns.barplot(x=share_freedom,y=region_lists,color='y',label="Freedom")
sns.barplot(x=share_trust,y=region_lists,color='r',label="Trust")
ax.legend(loc="lower right",frameon = True)
ax.set(xlabel='Percentage of Region', ylabel='Region',title = "Factors affecting happiness score")
plt.show()
sorted_data['region_happiness_ratio']=sorted_data['region_happiness_ratio']/max(sorted_data['region_happiness_ratio'])
sorted_data_economy['region_economy_ratio']=sorted_data_economy['region_economy_ratio']/max(sorted_data_economy['region_economy_ratio'])

data=pd.concat([sorted_data,sorted_data_economy['region_economy_ratio']],axis=1)
data.sort_values('region_happiness_ratio',inplace=True)

#Visualization
f,ax1 = plt.subplots(figsize =(20,10))
sns.pointplot(x='region',y='region_happiness_ratio',data=data,color='lime',alpha=0.8)
sns.pointplot(x='region',y='region_economy_ratio',data=data,color='red',alpha=0.8)
plt.text(7.55,0.6,'happiness score ratio',color='red',fontsize = 17,style = 'italic')
plt.text(7.55,0.55,'economy ratio',color='lime',fontsize = 18,style = 'italic')
plt.xticks(rotation=45)
plt.xlabel('Region',fontsize = 15,color='blue')
plt.ylabel('Values',fontsize = 15,color='blue')
plt.title('Happiness Score  VS  Economy Rate',fontsize = 20,color='blue')
plt.grid()
plt.show()
dataframe=pd.pivot_table(data_2015, index = 'Region', values=["Happiness_Score","Family"])
#to normalize
dataframe["Happiness_Score"]=dataframe["Happiness_Score"]/max(dataframe["Happiness_Score"])
dataframe["Family"]=dataframe["Family"]/max(dataframe["Family"])
sns.jointplot(dataframe.Family,dataframe.Happiness_Score,kind="kde",height=7,space=0)
plt.savefig('graph.png')
plt.show()
#Linear regression with marginal distributions
g = sns.jointplot("Family", "Happiness_Score", data=dataframe,height=5,kind="reg",ratio=3, color="r")
#broadcasting
data_2015['Year']=2015
data_2016['Year']=2016
data_2017['Year']=2017
#concating
data_concat=pd.concat([data_2015,data_2016,data_2017],axis=0,sort = False)

df=pd.pivot_table(data_concat, index = 'Year', values="Happiness_Score")
df
#pie chart
df.dropna(inplace = True)
labels =df.index
colors = ['cyan','lime','pink']
explode = [0,0,0]
sizes = df.values

# visual
plt.figure(figsize = (7,7))
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%')
plt.title('Percentage of year score by region',color = 'purple',fontsize = 15)
plt.show()
dataframe2=pd.pivot_table(data_2015, index = 'Region', values=["Happiness_Score","Trust"])
#to normalize
dataframe2["Happiness_Score"]=dataframe2["Happiness_Score"]/max(dataframe2["Happiness_Score"])
dataframe2["Trust"]=dataframe2["Trust"]/max(dataframe2["Trust"])
sns.lmplot("Trust","Happiness_Score",data=dataframe2)
plt.show()
dataframe3=pd.pivot_table(data_concat, index = 'Year', values=["Happiness_Score","Freedom"])
#to normalize
dataframe3["Happiness_Score"]=dataframe3["Happiness_Score"]/max(dataframe3["Happiness_Score"])
dataframe3["Freedom"]=dataframe3["Freedom"]/max(dataframe3["Freedom"])
sns.kdeplot(dataframe3.Freedom,dataframe3.Happiness_Score,shade=False,cut=5)
plt.show()
pal=sns.cubehelix_palette(2,rot=.5,dark=.3)
sns.violinplot(data=dataframe2, palette=pal, inner="points")
plt.show()
f,ax=plt.subplots(figsize=(8,8))
sns.heatmap(data_concat.corr(),annot=True,linewidth=.5,fmt='.1f',ax=ax)
plt.show()
f,ax = plt.subplots(figsize =(20,10))
sns.boxplot(x="Year" , y="Happiness_Score", hue="Region",data=data_concat,palette="PRGn",ax=ax)
plt.show()
f,ax = plt.subplots(figsize =(20,10))
sns.swarmplot(x="Year" , y="Happiness_Score", hue="Region",data=data_concat,ax=ax)
plt.show()
#Pair Plot
sns.pairplot(data)
plt.show()
f,ax = plt.subplots(figsize =(10,10))
sns.countplot(data_concat.Region,ax=ax)
plt.xticks(rotation= 45)
plt.show()
dataframe=data_concat.loc[:,["Year","Happiness_Score","Economy","Family","Health"]]
# Make the plot
plt.figure(figsize=(15,10))
parallel_coordinates(dataframe, 'Year', colormap=plt.get_cmap("Set1"))
plt.title(" visualization according to year features (2016, 2017, 2018)")
plt.xlabel("Features of data set")
plt.savefig('graph.png')
plt.show()