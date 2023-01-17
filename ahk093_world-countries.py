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

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
world = pd.read_csv("../input/countries of the world.csv")
world.head()
world.info()
world.fillna(0.0, inplace = True)
world.head()
def convert_currency(val):
    new_val = val.replace(',','.')
    return float(new_val)
world["Infant mortality (per 1000 births)"] = world["Infant mortality (per 1000 births)"].astype(str)
world["Infant mortality (per 1000 births)"] = world["Infant mortality (per 1000 births)"].apply(convert_currency)
area_list = list(world['Region'].unique())
mortality_ratio = []
for i in area_list:
    x = world[world['Region']==i]
    mortality_rate = sum(x["Infant mortality (per 1000 births)"])/len(x)
    mortality_ratio.append(mortality_rate)
data = pd.DataFrame({'area_list': area_list,'area_mortality_ratio':mortality_ratio})
new_index = (data['area_mortality_ratio'].sort_values(ascending=False)).index.values
sorted_data = data.reindex(new_index)
plt.figure(figsize=(15,10))
sns.barplot(x=sorted_data['area_list'], y=sorted_data['area_mortality_ratio'])
plt.xticks(rotation= 45)
plt.xlabel('Regions')
plt.ylabel('Mortality Birth Rate')
plt.title('Mortality Birth Rate Given States')
world['Net migration'] = world['Net migration'].astype(str)
world['Net migration'] = world['Net migration'].apply(convert_currency)
world['Net migration']
country_list = list(world['Country'].unique())
migration_ratio = []
for i in country_list:
    x = world[world['Country']==i]
    migration_rate = sum(x["Net migration"])/len(x)
    migration_ratio.append(migration_rate)
data2 = pd.DataFrame({'country_list': country_list,'country_migration_ratio':migration_ratio})
new_index = (data2['country_migration_ratio'].sort_values(ascending=False)).index.values
sorted_data2 = data2.reindex(new_index)
plt.figure(figsize=(15,10))
sns.barplot(x=sorted_data2['country_list'][:20], y=sorted_data2['country_migration_ratio'][:20])
plt.xticks(rotation= 45)
plt.xlabel('Countries')
plt.ylabel('Migration Rate')
plt.title('Migration Birth Rate Given Countries')
plt.figure(figsize=(15,10))
sns.barplot(x=sorted_data2['country_list'][-20:], y=sorted_data2['country_migration_ratio'][-20:])
plt.xticks(rotation= 45)
plt.xlabel('Countries')
plt.ylabel('Migration Rate')
plt.title('Migration Birth Rate Given Countries')
world.columns
world["Deathrate"] = world["Deathrate"].astype(str)
world["Deathrate"] = world["Deathrate"].apply(convert_currency)
world["GDP ($ per capita)"] = world["GDP ($ per capita)"].astype(str)
world["GDP ($ per capita)"] = world["GDP ($ per capita)"].apply(convert_currency)
region_list = list(world['Region'].unique())
deathrate = []
GDP = []
for i in region_list:
    x = world[world['Region']==i]
    deathrate1 = sum(x["Deathrate"])/len(x)
    gdp = sum(x["GDP ($ per capita)"])/len(x)
    deathrate.append(deathrate1)
    GDP.append(gdp)
data3 = pd.DataFrame({'region_list': region_list,'deathrate':deathrate})
new_index1 = (data3['deathrate'].sort_values(ascending=False)).index.values
sorted_data3 = data3.reindex(new_index1)
data4 = pd.DataFrame({'region_list': region_list,'GDP':GDP})
new_index2 = (data4['GDP'].sort_values(ascending=False)).index.values
sorted_data4 = data4.reindex(new_index2)
sorted_data3['deathrate'] = sorted_data3['deathrate']/max(sorted_data3['deathrate'])
sorted_data4['GDP'] = sorted_data4['GDP']/max(sorted_data4['GDP'])
data_new = pd.concat([sorted_data3,sorted_data4['GDP']],axis = 1)
data_new.sort_values('deathrate',inplace=True)

data_new
f,ax1 = plt.subplots(figsize =(20,10))
sns.pointplot(x='region_list',y='deathrate',data=data_new,color='lime',alpha=0.8)
sns.pointplot(x='region_list',y='GDP',data=data_new,color='red',alpha=0.8)
plt.text(40,0.6,'GDP',color='red',fontsize = 17,style = 'italic')
plt.text(40,0.55,'deathrate',color='lime',fontsize = 18,style = 'italic')
plt.xlabel('Regions',fontsize = 15,color='blue')
plt.ylabel('Values',fontsize = 15,color='blue')
plt.title('Deathrate  VS  GDP',fontsize = 20,color='blue')
plt.grid()
data_new.corr()
g = sns.jointplot(data_new.deathrate, data_new.GDP, kind="kde", size=7)
plt.savefig('graph.png')
plt.show()
