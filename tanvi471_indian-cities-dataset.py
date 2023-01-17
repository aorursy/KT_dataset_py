%matplotlib inline

import pandas as pd

from matplotlib import pyplot as plt

import matplotlib

import seaborn as sns

matplotlib.style.use('ggplot')

import numpy as np
data = pd.read_csv("../input/Indian_cities.csv")

data.head(10)
sns.heatmap(data.isnull(), cbar=False)
gp1=data.groupby('state_name').sum()

population = gp1[['population_male','population_female','population_total']].sort_values('population_total',ascending=False).drop('population_total',axis=1)

population
ax =population.plot(kind='bar',stacked=True, figsize=(15,8))

ax.set_xlabel("State")

ax.set_ylabel("Population in 10^7")
gp2=data.groupby('state_name').mean()

literacy = gp2[['effective_literacy_rate_total']].sort_values('effective_literacy_rate_total', ascending=True)                                              

literacy
ax1 =literacy.plot(kind='bar', figsize=(15,8),color='g')

ax1.set_xlabel("State")

ax1.set_ylabel("Effective literacy rate")
gp3=data.groupby('name_of_city').mean()

literacy_city = gp3[['effective_literacy_rate_total']].sort_values('effective_literacy_rate_total', ascending=False)                                              

literacy_city.head(20)
ax1 = literacy_city.head(20).plot(kind='bar', figsize=(15,8),color='b')

ax1.set_xlabel("State")

ax1.set_ylabel("Effective literacy rate")
ax1 = literacy_city.tail(20).plot(kind='bar', figsize=(15,8),color='b')

ax1.set_xlabel("State")

ax1.set_ylabel("Effective literacy rate")
gp4=data.groupby('state_name').mean()

sex_ratio = gp4[['sex_ratio']].sort_values('sex_ratio', ascending=False)                                              

sex_ratio.head(20)
ax1 = sex_ratio.plot(kind='bar', figsize=(15,8),color='b')

ax1.set_xlabel("State")

ax1.set_ylabel("Sex Ratio")
gp1=data.groupby('state_name').sum()

graduates = gp1[['total_graduates','male_graduates','female_graduates']].sort_values('total_graduates',ascending=False).drop('total_graduates',axis=1)

graduates
ax =graduates.plot(kind='bar',stacked=True, figsize=(18,7),fontsize =10)

ax.set_xlabel("State",size=10)

ax.set_ylabel("Graduate Count",size=10)
gp1=data.groupby('state_name').sum()

kid_population = gp1[['0-6_population_total','0-6_population_male','0-6_population_female']].drop('0-6_population_total',axis=1)

kid_population

ax =kid_population.plot(kind='bar',stacked=True, figsize=(18,7),fontsize =10)

ax.set_xlabel("State",size=10)

ax.set_ylabel("Kid Population",size=10)
grouped = data.groupby(['state_name']).nunique()

dist_count = pd.DataFrame(grouped['dist_code'])

district = dist_count.sort_values('dist_code')

district
ax =district.plot(kind='bar',stacked=True, figsize=(18,7),fontsize =10)

ax.set_xlabel("State",size=10)

ax.set_ylabel("Dist_count",size=10)
sns.set()

cols = ['population_total', '0-6_population_total', 'literates_total', 'sex_ratio', 'effective_literacy_rate_total', 'total_graduates']

sns.pairplot(data[cols], size = 2.5)

plt.show();



cols = ['population_total', '0-6_population_total', 'literates_total', 'sex_ratio', 'effective_literacy_rate_total', 'total_graduates']

corrmat = data[cols].corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmax=1, square=True);
k = 6 #number of variables for heatmap

cols = corrmat.nlargest(k, 'population_total')['population_total'].index

cm = np.corrcoef(data[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
from mpl_toolkits.basemap import Basemap

from numpy import array

from matplotlib import cm
data['latitude'] = data['location'].apply(lambda x: x.split(',')[0])

data['longitude'] = data['location'].apply(lambda x: x.split(',')[1])

data.head(10)
top_pop_cities = data.sort_values(by='population_total',ascending=False)

top10_pop_cities=top_pop_cities.head(10)

top10_pop_cities
plt.subplots(figsize=(20, 15))

map = Basemap(width=1200000,height=900000,projection='lcc',resolution='l',

                    llcrnrlon=67,llcrnrlat=5,urcrnrlon=99,urcrnrlat=37,lat_0=28,lon_0=77)



map.drawmapboundary ()

map.drawcountries ()

map.drawcoastlines ()



lg=array(top10_pop_cities['longitude'])

lt=array(top10_pop_cities['latitude'])

pt=array(top10_pop_cities['population_total'])

nc=array(top10_pop_cities['name_of_city'])



x, y = map(lg, lt)

population_sizes = top10_pop_cities["population_total"].apply(lambda x: int(x / 5000))

plt.scatter(x, y, s=population_sizes, marker="v", c=population_sizes, cmap=cm.Dark2, alpha=0.7)





for ncs, xpt, ypt in zip(nc, x, y):

    plt.text(xpt+70000, ypt+35000, ncs, fontsize=10, fontweight='bold')



plt.title('Top 10 Populated Cities in India',fontsize=20)
top10_least_literate_cities = data.sort_values(by='effective_literacy_rate_total',ascending=False)

top10_least_literate_cities=top10_least_literate_cities.tail(10)

top10_least_literate_cities
plt.subplots(figsize=(20, 15))

map = Basemap(width=1200000,height=900000,projection='lcc',resolution='l',

                    llcrnrlon=67,llcrnrlat=5,urcrnrlon=99,urcrnrlat=37,lat_0=28,lon_0=77)



map.drawmapboundary ()

map.drawcountries ()

map.drawcoastlines ()



lg=array(top10_least_literate_cities['longitude'])

lt=array(top10_least_literate_cities['latitude'])

pt=array(top10_least_literate_cities['literates_total'])

nc=array(top10_least_literate_cities['name_of_city'])



x, y = map(lg, lt)

population_sizes = top10_least_literate_cities["literates_total"].apply(lambda x: int(x / 5000))

plt.scatter(x, y, s=population_sizes, marker="v", c=population_sizes, cmap=cm.Dark2, alpha=0.7)





for ncs, xpt, ypt in zip(nc, x, y):

    plt.text(xpt+70000, ypt+35000, ncs, fontsize=10, fontweight='bold')



plt.title('Cities with the lowest literacy rate in India',fontsize=20)