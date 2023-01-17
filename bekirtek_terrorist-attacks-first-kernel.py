#import library 

import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns 

import os

print(os.listdir("../input"))
#read data  

terror = pd.read_csv('../input/globalterrorismdb_0718dist.csv',encoding='ISO-8859-1')
#Rename module and First 100 Terrorist Attacks 

terror.rename(columns={'iyear':'Year','imonth':'Month','iday':'Day','country_txt':'Country','region_txt':'Region','attacktype1_txt':'AttackType','target1':'Target','nkill':'Killed','nwound':'Wounded','summary':'Summary','gname':'Group','targtype1_txt':'Target_type','weaptype1_txt':'Weapon_type','motive':'Motive'},inplace=True) #rename

terror=terror[['Year','Month','Day','Country','Region','city','AttackType','Killed','Wounded','Target','Summary','Group','Target_type','Weapon_type','Motive']] #select column

terror['Casualities']=terror['Killed']+terror['Wounded'] #Total

terror.head(100) #First 100 Terrorist Attacks 
terror['Country'].value_counts() #Total attacks for country
terror['Region'].value_counts() #Total attacks for region 
terror['Country'].value_counts().index[0] #Highest attack for country
terror['Region'].value_counts().index[0] #Highest attack for region 
print('Max kill for country: '+terror.loc[terror['Killed'].idxmax()].Country+' / '+'Total: '+str(terror['Killed'].max())+' / '+'Group: '+terror.loc[terror['Killed'].idxmax()].Group+' / '+'Year: '+str(terror.loc[terror['Killed'].idxmax()].Year))

print(terror.AttackType.value_counts(dropna=False)) #  df.(dropna=False) - Remove missing values.
terror['Target_type'].value_counts()[0:10] #Top 10 Target
terror['Group'].value_counts()[1:11] #Top 10 Most Dangerous Terrorist Group Known 
#DataFrame.isin(values)[source]

#Whether each element in the DataFrame is contained in values.



top10group=terror[terror['Group'].isin(terror['Group'].value_counts()[1:11].index)]

pd.crosstab(top10group.Year,top10group.Group).plot(color=sns.color_palette('Paired',10))

fig=plt.gcf()

fig.set_size_inches(15,5)

plt.show() 



plt.subplots(figsize=(15,5))

sns.barplot(terror['Country'].value_counts()[:10].index,terror['Country'].value_counts()[:10].values)

plt.title('Top Affected Countries')

plt.show()
terror.tail(100) #Last 100 attack