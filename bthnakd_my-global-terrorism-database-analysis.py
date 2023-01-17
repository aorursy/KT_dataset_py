# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/globalterrorismdb_0718dist.csv',encoding='ISO-8859-1')
data.info()
data.corr()
data.head(10)
data.columns
data.rename(columns={'iyear':'Year','imonth':'Month','iday':'Day','country_txt':'Country','region_txt':'Region','attacktype1_txt':'AttackType','target1':'Target','nkill':'Killed','nwound':'Wounded','summary':'Summary','gname':'Group','targtype1_txt':'Target_type','weaptype1_txt':'Weapon_type','motive':'Motive'},inplace=True)
data.head(10)
data['casualities'] = data['Killed'] + data['Wounded']
data
type (data)
data ['Country']

data['Group'].value_counts().head (20)
print(data.describe())
#Number Of Terrorist Activities By Region
plt.subplots(figsize=(15,5))
sns.countplot('Region',data=data,palette='GnBu_d',edgecolor=sns.color_palette('dark',7),order=data['Region'].value_counts().index)
plt.xticks(rotation=90)
plt.title('Number Of Terrorist Activities By Region')
plt.show()
#Number Of Terrorist Activities Each Year
plt.subplots(figsize=(15,5))
sns.countplot('Year',data=data, palette='Greens',edgecolor=sns.color_palette('dark',7))
plt.xticks(rotation=90)
plt.title('Number Of Terrorist Activities Each Year')
plt.show()
#Attacking Methods
plt.subplots(figsize=(15,6))
sns.countplot('AttackType',data=data,palette='gist_earth_r',order=data['AttackType'].value_counts().index)
plt.xticks(rotation=90)
plt.title('Attacking Methods')
plt.show()
#Target Types
plt.subplots(figsize=(15,6))
sns.countplot(data['Target_type'],palette='CMRmap_r',order=data['Target_type'].value_counts().index)
plt.xticks(rotation=90)
plt.title('Target Types')
plt.show()
#Attacks Distribution Of Regions By Year
data_region=pd.crosstab(data.Year,data.Region)
data_region.plot(color=sns.color_palette('Set1',12))
fig=plt.gcf()
fig.set_size_inches(20,15)
plt.show()
#Attack types and regions
pd.crosstab(data.Region,data.AttackType).plot.barh(stacked=True,width=1,color=sns.color_palette('coolwarm',10))
fig=plt.gcf()
fig.set_size_inches(15,10)
plt.show()
#The Countries Most Affected By Terrorism
plt.subplots(figsize=(20,10))
sns.barplot(data['Country'].value_counts()[:10].index,data['Country'].value_counts()[:10].values,palette='cividis')
plt.title('Most Affected Countries')
plt.show()
#The Most Active Terror Groups
sns.barplot(data['Group'].value_counts()[:5].values,data['Group'].value_counts()[:5].index,palette=('cividis_r'))
plt.xticks(rotation=0)
fig=plt.gcf()
fig.set_size_inches(20,10)
plt.title('The Most Active Terror Groups')
plt.show()
#Top 10
top10groups=data[data['Group'].isin(data['Group'].value_counts()[:10].index)]
pd.crosstab(top10groups.Year,top10groups.Group).plot(color=sns.color_palette('tab20b',10))
fig=plt.gcf()
fig.set_size_inches(20,15)
plt.show()