# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
#read data to data frame
data = pd.read_csv('/kaggle/input/gtd/globalterrorismdb_0718dist.csv',encoding = "ISO-8859-1")
# change columns names to simplify
data.rename(columns={'iyear':'Year','imonth':'Month','iday':'Day','country_txt':'Country','region_txt':'Region','attacktype1_txt':'AttackType','target1':'Target','nkill':'Killed','nwound':'Wounded','summary':'Summary','gname':'Group','targtype1_txt':'Target_type','weaptype1_txt':'Weapon_type','motive':'Motive'},inplace=True)
df=data[['Year','Month','Day','Country','Region','city','latitude','longitude','AttackType','Killed','Wounded','Target','Summary','Group','Target_type','Weapon_type','Motive']]

df.info() #info about data
df.columns #info about columns
df.tail() #first 5 rows
df.corr() #correlation between columns
# correlation map
sns.set(font_scale=1.5)
f,ax = plt.subplots(figsize = (15,15))
sns.heatmap(df.corr(), annot = True, linewidths = .5, fmt = '.1f', ax=ax)
plt.show()
#Line Plot
df.Year.plot(kind = 'line', color = 'g', label = 'Year',linewidth=1, alpha= .5, grid = True, linestyle = '-.',figsize=(12,12))
plt.legend(loc='upper right')
plt.xlabel('Terror')
plt.ylabel('Year')
plt.title('Terror-Year Line Plot')
plt.show()
#Scatter Plot

plt.subplots(figsize=(12,12))
sns.scatterplot(data=df,x='Year',y='Killed',alpha=.5)
plt.title('Year-Killed Scatter Plot')
plt.show()
#Histogram Plot
df.Year.plot(kind = 'hist', bins = 100, figsize = (20,12))
plt.title('Number of Terror Activities Each Year')
plt.show()
# Count Plot
plt.subplots(figsize=(12,12))
sns.countplot('Region',data=df,palette='RdYlGn_r',edgecolor=sns.color_palette('dark',7),order=df['Region'].value_counts().index)
plt.xticks(rotation=90)
plt.title('Number Of Terrorist Activities Each Region')
plt.show()
terrorInTurkey = df[(df['Country'] == 'Turkey')] #terror data in Turkey
terrorInTurkey
# Count Plot

plt.subplots(figsize=(35,10))
sns.countplot('Group',data=terrorInTurkey,palette='RdYlGn_r',edgecolor=sns.color_palette('dark',7))
plt.xticks(rotation=90)
plt.title('Number Of Terror Group Activities in Turkey')
plt.show()
terrorInIstanbul = df[(df['city']=='Istanbul')&(df['Year']>2000)] #terror data after 2000 in Istanbul
terrorInIstanbul
# Count Plot
plt.subplots(figsize=(20,10))
sns.countplot(terrorInIstanbul['AttackType'],palette='inferno',order=terrorInIstanbul['AttackType'].value_counts().index)
plt.xticks(rotation=90)
plt.title('Number of Attack Type Activities After 2000 in Istanbul')

plt.show()
# DICTIONARY
dic = {'Riot Games':'LoL','Rockstar Games':'GTA','Valve':'CSGO'}
print(dic.keys())
print(dic.values())
dic['Riot Games'] = 'LoR' #update entry
dic['CD Projekt'] = 'Witcher 3' #add new entyr
print(dic)
print('LoR' in dic) # check inlude or not
print('Riot Games' in dic)
dic.clear() #clear dictionary
print(dic)
#WHILE AND FOR LOOPS
i = 0
while i<10:
    print(i*i)
    i+=1
    
lis = [1,2,3,4,5]
for index,value in enumerate(lis):
    print('index:',index,'value:',value)

dic = {'Riot Games':'LoL','Rockstar Games':'GTA','Valve':'CSGO'}
for key, value in dic.items():
    print(key+':',value)
    
for index,value in terrorInIstanbul[['Killed']][0:5].iterrows():
    print(index,value)
x = [1,2,True,'Ahmet']
print(max(x))