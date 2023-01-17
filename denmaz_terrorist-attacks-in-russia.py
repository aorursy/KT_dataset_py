# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime
import re
import folium
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
filename = '../input/globalterrorismdb_0617dist.csv'
#with codecs.decode(filename, encoding='utf-8', errors='ignore') as fdata:
df = pd.read_csv(filename, na_values=0, encoding='ISO-8859-1', usecols=[0, 1, 2, 3, 8, 13, 14, 18, 29, 35, 39, 58, 64, 82, 98, 101])
df.rename(columns={'iyear':'Year','imonth':'Month','iday':'Day','country_txt':'Country','attacktype1_txt':'AttackType','target1':'Target','nkill':'Killed','nwound':'Wounded','summary':'Summary','gname':'Group','targtype1_txt':'Target_type','weaptype1_txt':'Weapon_type','motive':'Motive'},inplace=True)
df=df[['eventid','Year','Month','Day','Country', 'latitude','longitude','AttackType','Killed','Wounded','Target','Summary','Group','Target_type','Weapon_type','Motive']]

df["Day"][df["Day"] == 0] = 1
df["Month"][df["Month"] == 0] = 1
df["Date"] = pd.to_datetime(df[["Day", "Month", "Year"]])

df = df.dropna(subset=['latitude', 'longitude'], how='all') # Remove rows without location
df[['Killed', 'Wounded']] = df[['Killed', 'Wounded']].fillna(0) # Fill with 0
df['Target'] = df['Target'].fillna("No info")
df['Casualties']=df['Killed']+df['Wounded']
df.set_index(['Date'], inplace=True)
df = df.sort_index()
df_ru = df[df["Country"]=="Russia"] # Filter attacks in Russia
split_by_attack = df_ru[['Killed', 'Wounded','Casualties','AttackType']][df_ru['Casualties']>0].loc['2000':].groupby('AttackType').sum()\
                                .sort_values(by='Casualties', ascending=False).reset_index()
split_by_attack
f, ax = plt.subplots(figsize=(8, 6))

sns.set(style="whitegrid")

sns.set_color_codes("pastel")
sns.barplot(x="Casualties", y="AttackType", data=split_by_attack,
            label="Casualties", color='b')

sns.set_color_codes("muted")
sns.barplot(x="Killed", y="AttackType", data=split_by_attack,
            label="Killed", color='b')

ax.set_title("Number of killed and wounded by type terrorist attacks in Russia\n\
Period: 2000 - 2017\n", loc='left', fontsize=14)

ax.set_xlabel(xlabel='Casualties')
ax.set_ylabel(ylabel='Type of Attack')

style = dict(ha="left", va="center", fontsize=10)
for i in range(0, len(split_by_attack)):
    ax.text(split_by_attack['Casualties'].iloc[i]+20, i, split_by_attack['Casualties'].iloc[i].astype(int), **style)

ax.legend(ncol=2, loc="lower right", frameon=True)
sns.despine(left=True, bottom=True)
target_of_attacks = df_ru[['Killed', 'Wounded','Casualties','Target_type']][df_ru['Casualties']>0].loc['2000':].groupby('Target_type').sum()\
                                .sort_values(by='Casualties', ascending=False).reset_index()

target_of_attacks
f, ax = plt.subplots(figsize=(8, 6))

sns.set(style="whitegrid")

sns.set_color_codes("pastel")
sns.barplot(x="Casualties", y="Target_type", data=target_of_attacks,
            label="Casualties", color='b')

sns.set_color_codes("muted")
sns.barplot(x="Killed", y="Target_type", data=target_of_attacks,
            label="Killed", color='b')

ax.set_title("Number of killed and wounded by target of terrorist attacks in Russia\n\
Period: 2000 - 2017\n", loc='left', fontsize=14)

ax.set_xlabel(xlabel='Casualties')
ax.set_ylabel(ylabel='Target')

style = dict(ha="left", va="center", fontsize=10)
for i in range(0, len(target_of_attacks)):
    ax.text(target_of_attacks['Casualties'].iloc[i]+20, i, target_of_attacks['Casualties'].iloc[i].astype(int), **style)


ax.legend(ncol=2, loc="lower right", frameon=True)
sns.despine(left=True, bottom=True)
piv = df_ru.loc['2000':].pivot_table('Casualties', index='Target_type', columns='AttackType', fill_value=0, aggfunc='count')
f, ax = plt.subplots(figsize=(10, 8))

ax.set_title("Correlation between type and target of terrorist attacks in Russia\n\
Period: 2000 - 2017\n", loc='left', fontsize=15)

plt.xticks(rotation=45)
ax = sns.heatmap(piv, cmap="OrRd")
plt.show()
labels_color = {'Bombing/Explosion':"red", 'Armed Assault':"blue", 'Hostage Taking (Barricade Incident)':"green", "Assassination":"yellow"}

m = folium.Map(
    location = [65.129220, 99.322836],
    zoom_start=3,
    tiles="Stamen Terrain"
)

temp = df_ru[['latitude', 'longitude', 'AttackType', 'Summary']].loc['2005':].reset_index()
n_rows = len(temp)

for i in range(1, n_rows):
    popup = folium.Popup(temp['Summary'].iloc[i], parse_html=True)
    if temp['AttackType'].iloc[i] in labels_color:
        folium.Marker([temp['latitude'].iloc[i], temp['longitude'].iloc[i]], icon=folium.Icon(color=labels_color[temp['AttackType'].iloc[i]]), popup=popup).add_to(m)
    else:
        folium.Marker([temp['latitude'].iloc[i], temp['longitude'].iloc[i]], icon=folium.Icon(color="lightgray"), popup=popup).add_to(m)
        

    
temp = None
m
