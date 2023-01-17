# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
%matplotlib inline
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Read data
us_census = pd.read_csv('../input/us-census-demographic-data/acs2015_county_data.csv',encoding="windows-1252")
outbreaks = pd.read_csv('../input/foodborne-diseases/outbreaks.csv',encoding="windows-1252")
students11_mat = pd.read_csv('../input/student-alcohol-consumption/student-por.csv',encoding="windows-1252")
us_census.head() #Show first 5 data
us_census.info() 
us_census.County.value_counts
#You can show counts of data values
us_census["State"].unique() #We can show unique datas.
us_census.head()
region_list = list(us_census['State'].unique())
region_income_ratio = []
for i in region_list:
    x = us_census[us_census['State']==i]                      #Find the state have how many county
    region_income_rate = sum(x.Income)/len(x)                 #Then calculate sum of income ratio and divided to found above
    region_income_ratio.append(region_income_rate)            #You append to list the state
    
    
#sorting
#Sort the income ratio as from low to high
#If you change the ascending state as False,Sorting will change as from high to low
data = pd.DataFrame({'region_list': region_list,'region_income_ratio':region_income_ratio})
new_index = (data['region_income_ratio'].sort_values(ascending=True)).index.values
sorted_data = data.reindex(new_index)

# visualization
plt.figure(figsize=(20,10))
sns.barplot(x=sorted_data['region_list'], y=sorted_data['region_income_ratio'])
plt.xticks(rotation= 90)
plt.xlabel('Region')
plt.ylabel('Income Rate')
plt.title('Income Rate Given Region')
# Most common 10 county name                 
loc_list =us_census['County']                      
loc_count = Counter(loc_list)         
most_common_locations = loc_count.most_common(10)  
x,y = zip(*most_common_locations)
x,y = list(x),list(y)
plt.figure(figsize=(20,10))
ax= sns.barplot(x=x, y=y,palette =sns.hls_palette(14, l=.2, s=.7))
plt.xlabel('Name of County')
plt.ylabel('Frequency')
plt.title('Most common 10 Name of County')
region_list = list(us_census['State'].unique())
state_poverty_ratio = []
for i in region_list:
    x = us_census[us_census['State']==i]                        #Find the state have how many county
    state_poverty_rate = sum(x.Poverty)/len(x)                  #Then you find sum of poverty ratio and divided to found above
    state_poverty_ratio.append(state_poverty_rate)              #You append to list the state
    
   
#sorting    
data = pd.DataFrame({'region_list': region_list,'state_poverty_ratio':state_poverty_ratio})
new_index = (data['state_poverty_ratio'].sort_values(ascending=False)).index.values
sorted_data2 = data.reindex(new_index)

# visualization
plt.figure(figsize=(20,10))
sns.barplot(x=sorted_data2['region_list'], y=sorted_data2['state_poverty_ratio'])
plt.xticks(rotation= 90)
plt.xlabel('State')
plt.ylabel('Poverty Rate')
plt.title('Poverty Rate Given State')
#Horizontal Bar Plot
area_list = list(us_census['State'].unique())

#We create 5 empty list to keep each races
share_white = []
share_black = []
share_native_american = []
share_asian = []
share_hispanic = []

#Find the number of each races in the States
for i in area_list:
    x = us_census[us_census['State']==i]
    share_white.append(sum(x.White)/len(x))
    share_black.append(sum(x.Black) / len(x))
    share_native_american.append(sum(x.Native) / len(x))
    share_asian.append(sum(x.Asian) / len(x))
    share_hispanic.append(sum(x.Hispanic) / len(x))

# visualization
f,ax = plt.subplots(figsize = (9,15))
sns.barplot(x=share_white,y=area_list,color='green',alpha = 0.5,label='White' )
sns.barplot(x=share_black,y=area_list,color='blue',alpha = 0.7,label='African American')
sns.barplot(x=share_native_american,y=area_list,color='cyan',alpha = 0.6,label='Native American')
sns.barplot(x=share_asian,y=area_list,color='yellow',alpha = 0.6,label='Asian')
sns.barplot(x=share_hispanic,y=area_list,color='red',alpha = 0.6,label='Hispanic')

ax.legend(loc='lower right',frameon = True)     # legendlarin gorunurlugu
ax.set(xlabel='Percentage of Races', ylabel='States',title = "Percentage of State's Population According to Races ")
data.head()
#We use datas above used
#Then make the basic a normalization
sorted_data['region_income_ratio'] = sorted_data['region_income_ratio']/max( sorted_data['region_income_ratio'])
sorted_data2['state_poverty_ratio'] = sorted_data2['state_poverty_ratio']/max( sorted_data2['state_poverty_ratio'])
data = pd.concat([sorted_data,sorted_data2['state_poverty_ratio']],axis=1)
data.sort_values('region_income_ratio',inplace=True)

# visualize
f,ax1 = plt.subplots(figsize =(30,10))
sns.pointplot(x='region_list',y='region_income_ratio',data=data,color='lime',alpha=0.8)
sns.pointplot(x='region_list',y='state_poverty_ratio',data=data,color='red',alpha=0.8)
plt.text(2,0.9,'Income Ratio',color='red',fontsize = 20,style = 'italic')
plt.text(2,0.85,'Poverty',color='lime',fontsize = 20,style = 'italic')
plt.xlabel('States',fontsize = 20,color='black')
plt.xticks(rotation= 90)
plt.ylabel('Values',fontsize = 20,color='black')
plt.title('Income Rate  VS  Poverty Rate',fontsize = 20,color='black')
plt.grid()
# Joint Plot
# Visualization of region income rate vs state poverty rate of each state with different style of seaborn code# joint kernel density
g = sns.jointplot(data.state_poverty_ratio, data.region_income_ratio, kind="kde", size=7)
plt.savefig('graph.png')
plt.show()
# you can change parameters of joint plot
# We use same datas but we show the technique another way
g = sns.jointplot("region_income_ratio", "state_poverty_ratio", data=data,size=5, ratio=3, color="b")
#We use outbreaks dataset and show the event happen each months
labels = outbreaks.Month.value_counts().index
colors = ['blue','grey','red','yellow','purple','beige','orange','green','brown','burlywood','pink','turquoise']
explode = [0,0,0,0,0,0,0,0,0,0,0,0]
sizes = outbreaks.Month.value_counts().values

# visual
plt.figure(figsize = (7,7))
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%')
plt.title('Event Happen According to Months',color = 'blue',fontsize = 15)
data.head()
# Visualization of poverty rate vs şncome rate of each state with different style of seaborn code
# Lmplot 
# Show the results of a linear regression within each dataset
sns.lmplot(x="region_income_ratio", y="state_poverty_ratio", data=data)
plt.show()
# Violin Plot
# Show each distribution with both violins and points
# Use cubehelix to get a custom sequential palette
pal = sns.cubehelix_palette(2, rot=-.5, dark=.3)
sns.violinplot(data=data, palette=pal, inner="points")
plt.show()

#Heatmap
# Visualization of  Region income rate vs Poverty rate of each state with different style of seaborn code
f,ax = plt.subplots(figsize=(5,5))
sns.heatmap(data.corr(), annot=True, linewidths=1,linecolor="blue", fmt= '.1f',ax=ax)
plt.show()
students11_mat=students11_mat.dropna()
students11_mat.head()
students11_mat.Pstatus.unique()
# Box Plot
# Pstatus(Parent's cohabitation status ) : A = living apart ,T=living together
# sex = F=female , M=male
# age
# Plot the orbital period with horizontal boxes
sns.boxplot(x="sex", y="age", hue="Pstatus",data=students11_mat, palette="PRGn")
plt.show()
# Swarm plot
# Pstatus(Parent's cohabitation status ) : A = living apart ,T=living together
# sex = F=female , M=male
# age
sns.swarmplot(x="sex", y="age", hue="Pstatus", data=students11_mat)
plt.show()
data.head()
# Count plot
# students alcohol
sns.countplot(students11_mat.sex)
plt.title("sex",color = 'blue',fontsize=15)

#If you want to see Pstatus you close the comment in this state
#sns.countplot(students11_mat.Pstatus)
#plt.title("Pstatus",color = 'blue',fontsize=15)




