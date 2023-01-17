import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
plt.style.use('fivethirtyeight')
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as g
import plotly.tools as tls
terror=pd.read_csv('../input/globalterrorismdb_0617dist.csv',encoding='ISO-8859-1',low_memory=False)
terror.rename(columns={'iyear':'Year','imonth':'Month','iday':'Day','country_txt':'Country','region_txt':'Region','attacktype1_txt':'AttackType','target1':'Target','nkill':'Killed','nwound':'Wounded','summary':'Summary','gname':'Group','targtype1_txt':'Target_type','weaptype1_txt':'Weapon_type','motive':'Motive'},inplace=True)
terror=terror[['Year','Month','Day','Country','Region','city','latitude','longitude','AttackType','Killed','Wounded','Target','Summary','Group','Target_type','Weapon_type','Motive']]
terror['casualities']=terror['Killed']+terror['Wounded']
terror.head(3)
print('Country with Highest Terrorist Attacks:',terror['Country'].value_counts().index[0])
print('Regions with Highest Terrorist Attacks:',terror['Region'].value_counts().index[0])
print('Month with Highest Terrorist Attacks:',terror['Month'].value_counts().index[0])
print('Most common attack type:',terror['AttackType'].value_counts().index[0])
print('Maximum people killed in an attack are:',terror['Killed'].max(),'that took place in',terror.loc[terror['Killed'].idxmax()].Country)
plt.subplots(figsize=(15,6))
sns.countplot('Year',data=terror,palette='RdYlGn_r',edgecolor=sns.color_palette('dark',7))
plt.xticks(rotation=90)
plt.title('Number of Terror Activities')
plt.show()
plt.subplots(figsize=(18,6))
sns.barplot(terror['Country'].value_counts()[:10].index,terror['Country'].value_counts()[:10].values,palette='inferno')
plt.title('Top Affected Countries - Number of Attacks')
plt.show()
coun_terror=terror['Country'].value_counts()[:15].to_frame()
coun_terror.columns=['Attacks']
coun_kill=terror.groupby('Country')['Killed'].sum().to_frame()
coun_terror.merge(coun_kill,left_index=True,right_index=True,how='left').plot.bar(width=0.9)
fig=plt.gcf()
fig.set_size_inches(18,6)
plt.show()
coun_terrorregion=terror['Region'].value_counts()[:15].to_frame()
coun_terrorregion.columns=['Attacks']
coun_killregion=terror.groupby('Region')['Killed'].sum().to_frame()
coun_terrorregion.merge(coun_killregion,left_index=True,right_index=True,how='left').plot.bar(width=0.9)
fig=plt.gcf()
fig.set_size_inches(18,6)
plt.show()
terror_inbefore2000 = terror[terror.Year <= 2000]
terror_after2000 = terror[terror.Year > 2000]
coun_terror1=terror_inbefore2000['Country'].value_counts()[:15].to_frame()
coun_terror1.columns=['Attacks']
coun_kill1=terror_inbefore2000.groupby('Country')['Killed'].sum().to_frame()
coun_terror1.merge(coun_kill1,left_index=True,right_index=True,how='left').plot.bar(width=0.9)
fig=plt.gcf()
fig.set_size_inches(12,6)
plt.ylim(0, 80000)
fig.suptitle('Terrorism Activities (1970-2000)', fontsize=16)
plt.tight_layout()
plt.show()

coun_terror2=terror_after2000['Country'].value_counts()[:15].to_frame()
coun_terror2.columns=['Attacks']
coun_kill2=terror_after2000.groupby('Country')['Killed'].sum().to_frame()
coun_terror2.merge(coun_kill2,left_index=True,right_index=True,how='left').plot.bar(width=0.9)
fig=plt.gcf()
fig.set_size_inches(12,6)
plt.ylim(0, 80000)
fig.suptitle('Terrorism Activities (2001-2016)', fontsize=16)
plt.tight_layout()
plt.show()
coun_terrorregion1=terror_inbefore2000['Region'].value_counts()[:15].to_frame()
coun_terrorregion1.columns=['Attacks']
coun_killregion1=terror_inbefore2000.groupby('Region')['Killed'].sum().to_frame()
coun_terrorregion1.merge(coun_killregion1,left_index=True,right_index=True,how='left').plot.bar(width=0.9)
fig=plt.gcf()
fig.set_size_inches(12,6)
plt.ylim(0, 80000)
fig.suptitle('Terrorism Activities by Region (1970-2000)', fontsize=16)
plt.tight_layout()
plt.show()

coun_terrorregion2=terror_after2000['Region'].value_counts()[:15].to_frame()
coun_terrorregion2.columns=['Attacks']
coun_killregion2=terror_after2000.groupby('Region')['Killed'].sum().to_frame()
coun_terrorregion2.merge(coun_killregion2,left_index=True,right_index=True,how='left').plot.bar(width=0.9)
fig=plt.gcf()
fig.set_size_inches(12,6)
plt.ylim(0, 80000)
fig.suptitle('Terrorism Activities by Region (2001-2016)', fontsize=16)
plt.tight_layout()
plt.show()
plt.subplots(figsize=(18,6))
sns.barplot(terror['Group'].value_counts()[1:15].index,terror['Group'].value_counts()[1:15].values,palette='inferno')
plt.xticks(rotation=90)
plt.title('Top Terror Group - By Number of Attacks')

plt.show()
coun_terrorgroup=terror['Group'].value_counts()[1:15].to_frame()
coun_terrorgroup.columns=['Attacks']
coun_killgroup=terror.groupby('Group')['Killed'].sum().to_frame()
coun_terrorgroup.merge(coun_killgroup,left_index=True,right_index=True,how='left').plot.bar(width=0.9)
fig=plt.gcf()
fig.set_size_inches(12,12)
fig.suptitle('Terrorism Activities by Group', fontsize=16)
plt.tight_layout()
plt.show()
coun_terrorattack=terror['AttackType'].value_counts()[:15].to_frame()
coun_terrorattack.columns=['Attacks']
coun_killattack=terror.groupby('AttackType')['Killed'].sum().to_frame()
coun_terrorattack.merge(coun_killattack,left_index=True,right_index=True,how='left').plot.bar(width=0.9)
fig=plt.gcf()
fig.set_size_inches(12,12)
fig.suptitle('Terrorism Activities by Attack Type', fontsize=16)
plt.tight_layout()
plt.show()