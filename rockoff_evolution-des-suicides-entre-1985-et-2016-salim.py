!pip install pycountry-convert
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import seaborn as sns

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



data = pd.read_csv("/kaggle/input/suicide-rates-overview-1985-to-2016/master.csv", index_col = "country")



data
data.dtypes
country_to_drop = ["Mongolia", "Macau", "Cabo Verde", "Dominica", "Bosnia and Herzegovina", "San Marino", "Saint Kitts and Nevis"]



data = data.drop(country_to_drop, axis=0)

data = data.reset_index()
data = data[data["year"] != 2016]

data["country"]=data["country"].replace('Republic of Korea', 'Korea, Republic of')

data["country"] = data["country"].replace('Saint Vincent and Grenadines', 'Saint Vincent and the Grenadines')

data["country"].unique()
import pycountry_convert as pc

data['continent'] = [pc.country_alpha2_to_continent_code(pc.country_name_to_country_alpha2(country, cn_name_format="default")) for country in data['country']]

data.head()

sns.set(rc={'figure.figsize':(30,30)}) # fonctionne pas 

# fig.set_size_inches(20, 20)



ax = sns.relplot(x="year", y="suicides/100k pop",ci=None,kind="line", data=data, height = 10).set(title = "Taux de suicide par an dans le monde")



# g.fig.autofmt_xdate()
sns.set_style()

sns.set_context('paper') 

ax = sns.barplot(x="year", y="suicides/100k pop", data=data)
sns.relplot(x="year", y="suicides/100k pop", hue="sex", ci=None, kind="line", data=data, height = 10).set(title = "Taux de suicide par an dans le monde par sexe")
plt.figure(figsize=(20,5))

sns.set_style()

sns.set_context('paper') 

ax = sns.barplot(y="suicides/100k pop", x="sex", data=data)
sns.relplot(x="year", y="suicides/100k pop", hue="age", ci=None, kind="line", data=data, height = 10).set(title = "Taux de suicide par an dans le monde en fonction de l'âge")
plt.figure(figsize=(20,8))

sns.set_style()

sns.set_context('talk') 

ax = sns.barplot(y="suicides/100k pop", x="age", order = ['75+ years','55-74 years','35-54 years','25-34 years','15-24 years','5-14 years'],data=data)
byAge = data.groupby(['sex', 'age']).agg(sum)

byAge = byAge.reset_index().sort_values(by = 'age', ascending=True)

fig = plt.figure(figsize=(15,4))

plt.title('Suicides by age and sex')

sns.barplot(y='suicides/100k pop', x='age', hue='sex', data=byAge, palette={'male': 'g', 'female': 'm'})


concap = pd.read_csv('../input/world-capitals-gps/concap.csv')

from mpl_toolkits.basemap import Basemap



by_country = data.groupby(['country']).agg(sum)



data_full = pd.merge(concap[['CountryName', 'CapitalName', 'CapitalLatitude', 'CapitalLongitude']], by_country, left_on='CountryName', right_on='country')

def mapWorld(col,size,title,label,metr=100,colmap='hot'):

    m = Basemap(projection='mill', llcrnrlat=-60, urcrnrlat=80, llcrnrlon=-130, urcrnrlon=190)

    m.drawcoastlines()

    m.drawcountries()

    



    lat = data_full['CapitalLatitude'].values

    lon = data_full['CapitalLongitude'].values

    a_1 = data_full[col].values

    if size:

        a_2 = data_full[size].values

    else: a_2 = 1



    m.scatter(lon, lat, latlon=True ,c=a_1, s=metr*a_2, linewidth=1, edgecolors='black', cmap=colmap)

    

    cbar = m.colorbar()

    cbar.set_label(label,fontsize=30)

    plt.title(title, fontsize=30)

    plt.show()

    

plt.figure(figsize=(20,10))

mapWorld(col='suicides_no', size=False, title='Suicides by countries', label='Nombres de suicides', metr=300, colmap='viridis')
# Remplacement du code continent par le nom

Continent_dict={'AF':'Africa', 'AS':'Asia', 'EU':'Europe', 'NA':'North America', 'OC':'Oceania', 'SA':'South America'}

data=data.replace(Continent_dict)



continent_group=data.groupby(['continent']).sum().reset_index()

continent_group.sort_values('suicides/100k pop', ascending=False)
fig, ax = plt.subplots(figsize=(15,7))

palette = sns.color_palette("Paired")

ax = sns.barplot(data=continent_group.sort_values('suicides/100k pop', ascending=False), x='continent', y='suicides/100k pop', ci=None, palette=palette)

ax.set_title('Suicides by continent (1985-2015)')
continent_time=data.groupby(['year','continent']).agg({'suicides/100k pop':sum}).reset_index()

continent_time

plt.figure(figsize=(20,10))

ax = sns.relplot(data=continent_time, x='year', y='suicides/100k pop', hue='continent',s=50)

ax.fig.suptitle('Suicide trends over time by continents (1985-2015)')
continent_time_wo_europe = continent_time[continent_time["continent"] != 'Europe']

continent_time_wo_europe
continent_time

#plt.figure(figsize=(20,20))

ax = sns.relplot(data=continent_time_wo_europe, x='year', y='suicides/100k pop', hue='continent',s=50)

ax.fig.suptitle('Suicide trends over time by continents excepting Europe (1985-2015)')
continent_sex=data.groupby(['sex','continent']).agg({'suicides/100k pop':sum}).reset_index()

continent_sex


# Set your custom color palette

colors = ["skyblue", "salmon"]

# Set your custom color palette

sns.set_palette(sns.color_palette(colors))

fig, ax = plt.subplots(figsize=(15,7))

ax = sns.barplot(data=continent_sex.sort_values('suicides/100k pop', ascending=False), x='continent', y='suicides/100k pop', hue='sex',palette=sns.set_palette(sns.color_palette(colors)))

ax.set_title('Suicides by continent and gender (1985-2015)')
continent_age=data.groupby(['age','continent']).agg({'suicides/100k pop':sum}).reset_index()

continent_age

palette=sns.color_palette("Paired", 9)

fig, ax = plt.subplots(figsize=(15,7))

ax = sns.barplot(data=continent_age.sort_values('suicides/100k pop', ascending=False), x='continent', y='suicides/100k pop', hue='age',palette=palette)

ax.set_title('Suicides by continent and age (1985-2015)')
top20=data.groupby(['country']).agg({'suicides/100k pop':sum}).reset_index().sort_values('suicides/100k pop', ascending=False)[0:20]

top20_list=[country for country in top20['country'].unique()]



 



data_top20=data[data['country'].isin(top20_list)]



 



data_top20_group_year=data_top20.groupby(['country','year']).agg({'suicides/100k pop':sum})



 



heatmap_data_year=data_top20_group_year.reset_index().pivot('country','year','suicides/100k pop')

#####



 



data_top20_group_age=data_top20.groupby(['country','age']).agg({'suicides/100k pop':sum})



 



heatmap_data_age=data_top20_group_age.reset_index().pivot('country','age','suicides/100k pop')



 





heatmap_data_age=heatmap_data_age.reindex(['5-14 years','15-24 years', '25-34 years', '35-54 years','55-74 years', '75+ years'], axis=1)

heatmap_data_age



 



fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(18, 7))

sns.heatmap(heatmap_data_year,cmap="YlGnBu", ax=axes[0])

sns.heatmap(heatmap_data_age,cmap="YlGnBu", ax=axes[1])

fig.suptitle('Top 20 countries based on suicide rates (suicides/100K) (1985-2015)', fontsize=16)
pd.isnull(data["HDI for year"]).value_counts()
data_hdi = data[~data["HDI for year"].isna()]
data_hdi
data_hdi = data_hdi.reset_index()

data_hdi
data_hdi_group = data_hdi.groupby(["country","year"])[["HDI for year","suicides/100k pop"]].mean().reset_index()
data_hdi_group
plt.figure(figsize=(20,10))

ax = sns.scatterplot(x="HDI for year", y="suicides/100k pop", data=data_hdi_group, hue = "year")

plt.axhline(data_hdi_group["suicides/100k pop"].mean(), label = "Moyenne")

plt.legend(loc='upper left')
g = sns.FacetGrid(data_hdi_group, col = "year", col_wrap=3)

g.map(plt.scatter, "HDI for year", "suicides/100k pop", alpha=.7)

g.add_legend()
pd.isnull(data["gdp_per_capita ($)"]).value_counts()
data_gdp = data

data_gdp
data_gdp_group = data_gdp.groupby(["country","year"])[["gdp_per_capita ($)","suicides/100k pop"]].mean().reset_index()

data_gdp_group
plt.figure(figsize=(20,10))

ax = sns.scatterplot(x="gdp_per_capita ($)", y="suicides/100k pop", data=data_gdp_group, hue = "gdp_per_capita ($)", legend = False)

plt.axhline(data_gdp_group["suicides/100k pop"].mean(), label = "Moyenne")

plt.legend(loc='upper left')
g = sns.FacetGrid(data_gdp_group, col = "year", col_wrap=5)

g.map(plt.scatter, "gdp_per_capita ($)", "suicides/100k pop", alpha=.7)

g.add_legend()