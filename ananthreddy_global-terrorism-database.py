import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
data = pd.read_csv('../input/globalterrorismdb_0617dist.csv', encoding = 'ISO-8859-1')
data.rename(columns={'iyear':'Year','imonth':'Month','iday':'Day','country_txt':'Country',
                     'region_txt':'Region','attacktype1_txt':'AttackType','target1':'Target',
                     'nkill':'Killed','nwound':'Wounded','summary':'Summary','gname':'Group',
                     'targtype1_txt':'Target_type','weaptype1_txt':'Weapon_type','motive':'Motive'}
                     ,inplace=True)
data = data[['Year','Month','Day','Country','Region','city','latitude','longitude','AttackType',
             'Killed','Wounded','Target','Summary','Group','Target_type','Weapon_type','Motive']]
data['casualities'] = data['Killed']+data['Wounded']
print ('Country with Highest Terrorist Attacks:',data['Country'].value_counts().index[0])
print ('Regions with Highest Terrorist Attacks:',data['Region'].value_counts().index[0])
df = (data.groupby(['Country']).sum())
print ('Country with Highest Casualities:',df.sort_values(by = ['casualities'],ascending=[False]).index[0])
plt.subplots(figsize=(15,6))
sns.countplot('Year',data=data,palette='RdYlGn_r',edgecolor=sns.color_palette('dark',7))
plt.xticks(rotation=90)
plt.title('Number Of Terrorist Activities Each Year')
plt.show()
df = (data.groupby(['Year']).sum())
print ("Year with Highest Number of Casualities : ",df.sort_values(by=['casualities'],ascending=[False])
       .index[0])

plt.subplots(figsize=(15,6))
sns.barplot(x = data['Year'].unique(), y = df['casualities'], data= df, palette='rainbow')
plt.xticks(rotation=90)
plt.title('Terror Casualities Year Wise')
plt.show()
plt.subplots(figsize=(15,6))
sns.barplot(x = data['Group'].value_counts()[1:15].values, y = data['Group'].value_counts()[1:15].index, palette = 'rainbow_r')
plt.subplots(figsize=(15,6))
data_group = data.groupby(['Group']).sum()
data_group = data_group.sort_values(by = ['casualities'], ascending = False)
sns.barplot (data_group['casualities'][1:15],data_group.index[1:15],palette = 'rainbow_r')
plt.subplots(figsize=(10,5))
sns.countplot(y = 'AttackType',data=data,palette='RdYlGn',order=data['AttackType'].value_counts().index)
plt.xticks(rotation=90)
plt.title('Attacking Methods by Terrorists')
plt.show()
data_india = data[data['Country']=='India']
data_india['casualities'] = data_india['Killed']+data_india['Wounded']
plt.subplots(figsize=(15,6))
sns.countplot('Year',data=data_india,palette='RdYlGn_r',edgecolor=sns.color_palette('dark',7))
plt.xticks(rotation=90)
plt.title('Number Of Terrorist Activities Each Year in India')
plt.show()
plt.subplots(figsize=(10,6))
ind_groups=data_india['Group'].value_counts()[1:15].index
ind_groups=data_india[data_india['Group'].isin(ind_groups)]
sns.countplot(y='Group', data=ind_groups, order = ind_groups['Group'].value_counts().index)
# plt.xticks(rotation=90)
plt.subplots(figsize=(15,6))
data_india_group = data_india.groupby(['Group']).sum()
data_india_group = data_india_group.sort_values(by = ['casualities'], ascending = False)
sns.barplot (data_india_group['casualities'][1:15],data_india_group.index[1:15],palette = 'rainbow_r')