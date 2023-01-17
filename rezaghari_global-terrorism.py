# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd
import numpy as np # linear algebra
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
plt.style.use('fivethirtyeight')
from mpl_toolkits.basemap import Basemap
import folium
import folium.plugins
from matplotlib import animation,rc
import io
import base64
from IPython.display import HTML, display
import warnings
warnings.filterwarnings('ignore')
import codecs

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/gtd/globalterrorismdb_0718dist.csv', encoding = "ISO-8859-1")
df.head(5)
df.rename(columns={'iyear':'Year','imonth':'Month','city':'City','iday':'Day','country_txt':'Country','region_txt':'Region','attacktype1_txt':'AttackType','target1':'Target','nkill':'Killed','nwound':'Wounded','summary':'Summary','gname':'Group','targtype1_txt':'Target_type','weaptype1_txt':'Weapon_type','motive':'Motive'},inplace=True)
df['Casualities'] = df.Killed + df.Wounded
df=df[['Year','Month','Day','Country','Region','City','latitude','longitude','AttackType','Killed','Wounded','Casualities','Target','Group','Target_type','Weapon_type']]

df.head()
# df.to_csv('global_terror_v2.csv')
df.isnull().sum()
print(f"""
    There are {df.Country.nunique()} countries from {df.Region.nunique()} regions covered in the dataset and terrorist atacks data in {df.Year.nunique()}
    years from {df.Year.min()} to {df.Year.max()}. Overally {df.index.nunique()} terrorist attacks are recorded here which caused about {int(df.Casualities.sum())} casualities
    consisted of {int(df.Killed.sum())} kills and {int(df.Wounded.sum())} wounded.
""")
print(f"The highest terrorist attacks were commited in {df.Country.value_counts().index[0]} with {df.Country.value_counts().max()} attacks")

print('The other 4 countries with highest terrorist attacks are:')
for i in range(1,5):
    print(f"{i+1}. {df.Country.value_counts().index[i]} with {df.Country.value_counts()[i]} attacks")
print(f"The region that highest terrorist attacks were commited in {df.Region.value_counts().index[0]} with {df.Region.value_counts().max()} attacks")

print('The other regions orderd by highest terrorist attacks are:')
for i in range(1,7):
    print(f"{i+1}. {df.Region.value_counts().index[i]} with {df.Region.value_counts()[i]} attacks")
plt.subplots(figsize=(15,6))
sns.countplot('Year',data=df,palette='RdYlGn_r',edgecolor=sns.color_palette('dark',10))
plt.xticks(rotation=90)
plt.title('Number Of Terrorist Activities Each Year')
plt.show()
plt.subplots(figsize=(15,6))
year_cas = df.groupby('Year').Casualities.sum().to_frame().reset_index()
year_cas.columns = ['Year','Casualities']
sns.barplot(x=year_cas.Year, y=year_cas.Casualities, palette='RdYlGn_r',edgecolor=sns.color_palette('dark',10))
plt.xticks(rotation=90)
plt.title('Number Of Casualities Each Year')
plt.show()
plt.subplots(figsize=(15,6))
country_attacks = df.Country.value_counts()[:15].reset_index()
country_attacks.columns = ['Country', 'Total Attacks']
sns.barplot(x=country_attacks.Country, y=country_attacks['Total Attacks'], palette= 'OrRd_r',edgecolor=sns.color_palette('dark',10))
plt.xticks(rotation=30)
plt.title('Number Of Total Attacks in Each Country')
plt.show()
plt.subplots(figsize=(15,6))
count_cas = df.groupby('Country').Casualities.sum().to_frame().reset_index().sort_values('Casualities', ascending=False)[:15]
sns.barplot(x=count_cas.Country, y=count_cas.Casualities, palette= 'OrRd_r',edgecolor=sns.color_palette('dark',10))
plt.xticks(rotation=30)
plt.title('Number Of Total Casualities in Each Country')
plt.show()
region_attacks = df.Region.value_counts().to_frame().reset_index()
region_attacks.columns = ['Region', 'Total Attacks']
plt.subplots(figsize=(15,6))
sns.barplot(x=region_attacks.Region, y=region_attacks['Total Attacks'], palette='OrRd_r', edgecolor=sns.color_palette('dark',10))
plt.xticks(rotation=90)
plt.title('Number Of Total Attacks in Each Region')
plt.show()
attack_type = df.AttackType.value_counts().to_frame().reset_index()
attack_type.columns = ['Attack Type', 'Total Attacks']
plt.subplots(figsize=(15,6))
sns.barplot(x=attack_type['Attack Type'], y=attack_type['Total Attacks'], palette='YlOrRd_r',
            edgecolor=sns.color_palette('dark', 10))
plt.xticks(rotation=90)
plt.title('Number Of Total Attacks by Attack Type')
plt.show()
baghdad_att = df.City[df.City=='Lima'].value_counts()
baghdad_cas = 76897

baghdad_att
city_attacks = df.City.value_counts().to_frame().reset_index()
city_attacks.columns = ['City', 'Total Attacks']
city_cas = df.groupby('City').Casualities.sum().to_frame().reset_index()
city_cas.columns = ['City', 'Casualities']
# city_cas.drop('Unknown', axis=0, inplace=True)
city_tot = pd.merge(city_attacks, city_cas, how='left', on='City').sort_values('Total Attacks', ascending=False)[1:21]
# fig = plt.figure()
# fig.subplots()
sns.set_palette('RdBu')
city_tot.plot.bar(x='City', width=0.8)
plt.xticks(rotation=90)
plt.title('Number Of Total Attacks and Casualities by City')
fig = plt.gcf()
fig.set_size_inches(16,9)
plt.show()

group_attacks = df.Group.value_counts().to_frame().drop('Unknown').reset_index()[:16]
group_attacks.columns = ['Terrorist Group', 'Total Attacks']
group_attacks
group_attacks = df.Group.value_counts().to_frame().drop('Unknown').reset_index()[:16]
group_attacks.columns = ['Terrorist Group', 'Total Attacks']
plt.subplots(figsize=(10,8))
sns.barplot(y=group_attacks['Terrorist Group'], x=group_attacks['Total Attacks'], palette='YlOrRd_r',
            edgecolor=sns.color_palette('dark', 10))
# plt.xticks()
plt.title('Number Of Total Attacks by Terrorist Group')
plt.show()
groups_10 = df[df.Group.isin(df.Group.value_counts()[1:11].index)]
pd.crosstab(groups_10.Year, groups_10.Group).plot(color=sns.color_palette('Paired', 10))
fig=plt.gcf()
fig.set_size_inches(18,6)
plt.xticks(range(1970, 2017, 5))
plt.ylabel('Total Attacks')
plt.title('Top Terrorist Groups Activities from 1970 to 2017')
plt.legend(labels=['Al-Shabaab',
                   'Boko Haraam',
                   'FMLN',
                   'IRA',
                   'ISIL',
                   'PKK',
                   'NPA',
                   'FARC',
                   'SL',
                   'Taliban'], loc='upper left')
plt.show()
terror_fol=df.copy()
terror_fol.dropna(subset=['latitude','longitude'],inplace=True)
location_fol=terror_fol[['latitude','longitude']][:5000]
country_fol=terror_fol['Country'][:5000]
city_fol=terror_fol['City'][:5000]
killed_fol=terror_fol['Killed'][:5000]
wound_fol=terror_fol['Wounded'][:5000]
def color_point(x):
    if x>=30:
        color='red'
    elif ((x>0 and x<30)):
        color='blue'
    else:
        color='green'
    return color   
def point_size(x):
    if (x>30 and x<100):
        size=2
    elif (x>=100 and x<500):
        size=8
    elif x>=500:
        size=16
    else:
        size=0.5
    return size   
map2 = folium.Map(location=[30,0],tiles='CartoDB dark_matter',zoom_start=2)
for point in location_fol.index:
    info='<b>Country: </b>'+str(country_fol[point])+'<br><b>City: </b>: '+str(city_fol[point])+'<br><b>Killed </b>: '+str(killed_fol[point])+'<br><b>Wounded</b> : '+str(wound_fol[point])
    iframe = folium.IFrame(html=info, width=200, height=200)
    folium.CircleMarker(list(location_fol.loc[point].values),popup=folium.Popup(iframe),radius=point_size(killed_fol[point]),color=color_point(killed_fol[point])).add_to(map2)
map2
def change_case(text):
    text = text.lower()
    return text[0].upper()+text[1:]
iran_attacks = df[df.Country=='Iran'].reset_index()
iran_attacks.City = iran_attacks.City.apply(change_case)
iran_attacks.head()
print(f"""
    A total number of {iran_attacks.City.nunique()} cities attacked by {iran_attacks.Group.nunique()} different terrorist groups covered in the dataset between
    {iran_attacks.Year.min()} to {iran_attacks.Year.max()}. Overally {iran_attacks.index.nunique()} terrorist attacks are recorded here which caused about {int(iran_attacks.Casualities.sum())} casualities
    consisted of {int(iran_attacks.Killed.sum())} kills and {int(iran_attacks.Wounded.sum())} wounded.
""")
iran_attacks.City.value_counts()
# iran_attacks.to_csv('iran_terror_v2.csv')
iran_attacks.isnull().sum()
print(f"The highest terrorist attacks were commited in {iran_attacks.City.value_counts().index[0]} with {iran_attacks.City.value_counts().max()} attacks")

print('The other 4 Iranian cities with highest terrorist attacks are:')
for i in range(2,6):
    print(f"{i}. {iran_attacks.City.value_counts().index[i]} with {iran_attacks.City.value_counts()[i]} attacks")
plt.subplots(figsize=(15,6))
sns.countplot('Year',data=iran_attacks,palette='RdYlGn_r',edgecolor=sns.color_palette('dark',10))
plt.xticks(rotation=90)
plt.title('Number Of Terrorist Activities in Iran Each Year')
plt.show()
plt.subplots(figsize=(15,6))
year_cas_iran = iran_attacks.groupby('Year').Casualities.sum().to_frame().reset_index()
year_cas_iran.columns = ['Year','Casualities']
sns.barplot(x=year_cas_iran.Year, y=year_cas_iran.Casualities, palette='RdYlGn_r',
            edgecolor=sns.color_palette('dark',10))
plt.xticks(rotation=90)
plt.title('Number Of Casualities in Iran Each Year')
plt.show()
plt.subplots(figsize=(15,6))
city_attacks_iran = iran_attacks.City.value_counts()[:15].reset_index()
city_attacks_iran.columns = ['City', 'Total Attacks']
city_attacks_iran.drop(1, inplace=True)
sns.barplot(x=city_attacks_iran.City, y=city_attacks_iran['Total Attacks'], palette='OrRd_r',
            edgecolor=sns.color_palette('dark',10))
plt.xticks(rotation=30)
plt.title('Number Of Total Attacks in Each Iranian City')
plt.show()
plt.subplots(figsize=(15,6))
city_attacks_iran = iran_attacks.City.value_counts()[:17].reset_index()
city_attacks_iran.columns = ['City', 'Total Attacks']
city_attacks_iran.drop([0, 1], inplace=True)
sns.barplot(x=city_attacks_iran.City, y=city_attacks_iran['Total Attacks'], palette='OrRd_r',
            edgecolor=sns.color_palette('dark',10))
plt.xticks(rotation=30)
plt.title('Number Of Total Attacks in Non-Capital Cities of Iran')
plt.show()
ir_attack_type = iran_attacks.AttackType.value_counts().to_frame().reset_index()
ir_attack_type.columns = ['Attack Type', 'Total Attacks']
plt.subplots(figsize=(15,6))
sns.barplot(x=ir_attack_type['Attack Type'], y=ir_attack_type['Total Attacks'], palette='YlOrRd_r',
            edgecolor=sns.color_palette('dark', 10))
plt.xticks(rotation=90)
plt.title('Number Of Total Attacks in Iran by Attack Type')
plt.show()
ir_city_attacks = iran_attacks.City.value_counts().to_frame().reset_index()
ir_city_attacks.columns = ['City', 'Total Attacks']
ir_city_cas = iran_attacks.groupby('City').Casualities.sum().to_frame().reset_index()
ir_city_cas.columns = ['City', 'Casualities']
# city_cas.drop('Unknown', axis=0, inplace=True)
ir_city_tot = pd.merge(ir_city_attacks, ir_city_cas, how='left', on='City').sort_values('Casualities',
                                                                                  ascending=False)[1:12]
# ir_city_tot.drop(1, inplace=True)
ir_city_tot
# fig = plt.figure()
# fig.subplots()
sns.set_palette('RdBu')
ir_city_tot.plot.bar(x='City', width=0.8)
plt.xticks(rotation=90)
plt.title('Number Of Total Attacks and Casualities by Each Non-Capital Iranian City')
fig = plt.gcf()
fig.set_size_inches(16,7)
plt.xticks(rotation=0)
plt.show()

ir_group_attacks = iran_attacks.Group.value_counts().to_frame().drop('Unknown').reset_index()[:8]
ir_group_attacks.columns = ['Terrorist Group', 'Total Attacks']
ir_group_attacks
plt.subplots(figsize=(10,8))
sns.barplot(y=ir_group_attacks['Terrorist Group'], x=ir_group_attacks['Total Attacks'], palette='YlOrRd_r',
            edgecolor=sns.color_palette('dark', 10))
plt.xticks(range(0,110,10))
plt.title('Number Of Total Attacks by Terrorist Group in Iran')
plt.show()
ir_groups_10 = iran_attacks[iran_attacks.Group.isin(iran_attacks.Group.value_counts()[1:9].index)]
pd.crosstab(ir_groups_10.Year, ir_groups_10.Group).plot(color=sns.color_palette('Paired', 10))
fig=plt.gcf()
fig.set_size_inches(18,6)
plt.xticks(range(1970, 2017, 5))
plt.ylabel('Total Attacks')
plt.title('Top Terrorist Groups Activities in Iran from 1970 to 2017')
plt.show()
terror_fol=iran_attacks.copy()
terror_fol.dropna(subset=['latitude','longitude'],inplace=True)
location_fol=terror_fol[['latitude','longitude']]
country_fol=iran_attacks.Country
city_fol=terror_fol['City']
killed_fol=terror_fol['Killed']
wound_fol=terror_fol['Wounded']
group_fol=terror_fol['Group']
def color_point(x):
    if x>=30:
        color='red'
    elif ((x>0 and x<30)):
        color='blue'
    else:
        color='green'
    return color   
def point_size(x):
    if (x>5 and x<30):
        size=2
    elif (x>=30 and x<100):
        size=6
    elif (x>=100 and x<250):
        size=12
    elif (x>=250 and x<500):
        size=18
    elif x>=500:
        size=30
    else:
        size=0.5
    return size   
map2 = folium.Map(location=[32.4279,53.6880],tiles='CartoDB dark_matter',zoom_start=6)
for point in location_fol.index:
    info='<b>City: </b>: '+str(city_fol[point])+'<br><b>Killed </b>: '+str(killed_fol[point])+'<br><b>Wounded</b> : '+str(wound_fol[point])+'<br><b>Group</b> : '+str(group_fol[point])
    iframe = folium.IFrame(html=info, width=200, height=200)
    folium.CircleMarker(list(location_fol.loc[point].values),popup=folium.Popup(iframe),
                        radius=point_size(killed_fol[point]),color=color_point(killed_fol[point])).add_to(map2)
    folium.TileLayer('MapQuest Open Aerial', attr="<a href=https://kaggle.com/rezaghari>Reza Ghari</a>").add_to(map2)
map2
fig = plt.figure(figsize = (10,8))
def animate(Year):
    ax = plt.axes()
    ax.clear()
    ax.set_title('Terrorism In Iran '+'\n'+'Year:' +str(Year))
    m5 = Basemap(projection='lcc',resolution='l',llcrnrlon=44,llcrnrlat=24,urcrnrlon=67,urcrnrlat=41,lat_0=32,lon_0=53)
    lat_gif=list(iran_attacks[iran_attacks['Year']==Year].latitude)
    long_gif=list(iran_attacks[iran_attacks['Year']==Year].longitude)
    x_gif,y_gif=m5(long_gif,lat_gif)
    m5.scatter(x_gif, y_gif,s=[killed+wounded for killed,wounded in zip(iran_attacks[iran_attacks['Year']==Year].Killed,iran_attacks[iran_attacks['Year']==Year].Wounded)],color = 'r')
    m5.drawcoastlines()
    m5.drawcountries()
    m5.fillcontinents(color='coral',lake_color='aqua', zorder = 1,alpha=0.4)
    m5.drawmapboundary(fill_color='aqua')
ani = animation.FuncAnimation(fig,animate,list(iran_attacks.Year.unique()), interval = 1500)    
ani.save('animation.gif', writer='imagemagick', fps=1)
plt.close(1)
filename = 'animation.gif'
video = io.open(filename, 'r+b').read()
encoded = base64.b64encode(video)
HTML(data='''<img src="data:image/gif;base64,{0}" type="gif" />'''.format(encoded.decode('ascii')))