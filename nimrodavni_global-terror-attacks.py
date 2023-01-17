# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import os

import io

import numpy as np

import pandas as pd

from IPython.display import Image

from matplotlib import pyplot as plt

import seaborn as sns





%matplotlib inline 

pd.options.mode.chained_assignment = None  # default='warn'
# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/globalterror/GlobalTerror.csv')
df.columns
df.head()
# Level = Easy

# Tests = mean

df['Casualities'].mean()
# Level: Easy / Medium

# Tests: Barplot

plt.subplots(figsize=(15,6))

sns.countplot('Year',data=df,palette='RdYlGn_r',edgecolor=sns.color_palette('dark',7))

plt.xticks(rotation=90) #make x ticks (years) horizontal' so they won't collapse with eachother

plt.title('Number Of Terrorist Activities Each Year')

plt.show()
# Level: Easy/Medium

# Tests: Horizontal Barplot, Filtering

plt.subplots(figsize=(15,6))

sns.countplot(y='TargetType',data=df[df['Region'] == 'North America'],orient ='horizontal',palette='inferno')

plt.title('Most common targets in the north american region')

plt.show()
# Level: Easy

# Tests: Piechart

df['WeaponType'].value_counts().plot.pie()
# Level: Medium

# Tests: Piechart, Working a series

weapon_series = df['WeaponType'].value_counts()

values_to_show = 5

weapon_series = weapon_series.nlargest(values_to_show).append(pd.Series(weapon_series.nsmallest(weapon_series.size - values_to_show).sum(),index=['other']))

weapon_series.plot.pie()
# Level: Really Hard

# Tests: Filtering, World Map, Filtering

from mpl_toolkits.basemap import Basemap

import matplotlib.patches as mpatches

m3 = Basemap(projection='mill',llcrnrlat=-80,urcrnrlat=80, llcrnrlon=-180,urcrnrlon=180,lat_ts=20,resolution='c',lat_0=True,lat_1=True)

lat_100=list(df[df['Casualities']>=75].latitude)

long_100=list(df[df['Casualities']>=75].longitude)

x_100,y_100=m3(long_100,lat_100)

m3.plot(x_100, y_100,'go',markersize=5,color = 'r')

lat_=list(df[df['Casualities']<75].latitude)

long_=list(df[df['Casualities']<75].longitude)

x_,y_=m3(long_,lat_)

m3.plot(x_, y_,'go',markersize=2,color = 'b',alpha=0.4)

m3.drawcoastlines()

m3.drawcountries()

m3.fillcontinents(lake_color='aqua')

m3.drawmapboundary(fill_color='aqua')

fig=plt.gcf()

fig.set_size_inches(10,6)

plt.title('Global Terrorist Attacks')

plt.legend(loc='lower left',handles=[mpatches.Patch(color='b', label = "< 75 casualities"),

                    mpatches.Patch(color='red',label='> 75 casualities')])

plt.show()
from IPython.display import HTML

import base64

gif = io.open('/kaggle/input/globalterror/UsaTerror.gif', 'rb').read()

encoded = base64.b64encode(gif)

HTML(data='''<img src="data:image/gif;base64,{0}" type="gif" />'''.format(encoded.decode('ascii')))
# Level: Crazy Hard

# Tests: Filtering, WorldMap, ScatterPlot, Gif

from matplotlib import animation,rc

from IPython.display import display

import warnings

terror_israel=df[df['Country']=='Israel']

fig = plt.figure(figsize = (10,8))

def animate(Year):

    ax = plt.axes()

    ax.clear()

    ax.set_title('Terrorism In Israel '+'\n'+'Year:' +str(Year))

    m6 = Basemap(projection='mill',llcrnrlat=28,llcrnrlon=34,urcrnrlat=35,urcrnrlon=37,lat_ts=20,resolution='c',lat_0=True,lat_1=True)

    lat_gif1=list(terror_israel[terror_israel['Year']==Year].latitude)

    long_gif1=list(terror_israel[terror_israel['Year']==Year].longitude)

    x_gif1,y_gif1=m6(long_gif1,lat_gif1)

    m6.scatter(x_gif1, y_gif1,s=[killed+wounded for killed,wounded in zip(terror_israel[terror_israel['Year']==Year].Killed,terror_israel[terror_israel['Year']==Year].Wounded)],color ='r') 

    m6.drawcoastlines()

    m6.drawcountries()

    m6.fillcontinents(color='white',lake_color='aqua', zorder = 1,alpha=0.4)

    m6.drawmapboundary(fill_color='aqua')

ani = animation.FuncAnimation(fig,animate,list(terror_israel.Year.unique()), interval = 1500)    

ani.save('animation.gif', writer='imagemagick', fps=1)

plt.close(1)

filename = 'animation.gif'

video = io.open(filename, 'r+b').read()

encoded = base64.b64encode(video)

HTML(data='''<img src="data:image/gif;base64,{0}" type="gif" />'''.format(encoded.decode('ascii')))
# Level: Easy

# Tests: Groupby, mean, barplot

casualities_per_region = df.groupby('Region')['Casualities'].sum()

sns.barplot(casualities_per_region.index,casualities_per_region.values)

plt.xticks(rotation=90) #make x ticks (years) horizontal' so they won't collapse with eachother

plt.title('Number of casualities per region')

plt.show()
# Level: Easy

# Tests: Filtering

list(df[(df['Day'] == 11) & (df['Month'] == 9) & (df['Year'] == 2001) & (df['Country'] == 'United States')]['Summary'])
# Level: Easy

# Tests: loc / iloc

df.loc[73905]['Summary']
# Level: Medium / Hard

# Tests: Boxplot, Filtering, Groupby

top_n = 10

top_n_killer_groups = list(df.groupby('Group')['Casualities'].sum().sort_values(ascending=False).nlargest(top_n).index)

sns.boxplot(x = "Group", y = "Casualities",data = df[df['Group'].isin(top_n_killer_groups)])

plt.xticks(rotation=90) #make x ticks (years) horizontal' so they won't collapse with eachother

plt.title('Number of casualities per region')

plt.ylim(0,150)

plt.show()
# Level: Very Easy

# Tests: sorting

most_deadly_attack = df.sort_values(by='Killed',ascending=False).max()

print(f'The most deadly terror attack was preformed in {most_deadly_attack["Country"]} by {most_deadly_attack["Group"]} and {int(most_deadly_attack["Killed"])} were killed')
# Level: Hard

# Tests: Lineplot, Groupby, Working with series

top_groups10=df['Group'].value_counts()[1:11].index

fig, ax = plt.subplots(figsize=(15,7))

df[df['Group'].isin(top_groups10)].groupby(['Year','Group'])['Group'].value_counts().droplevel(2).unstack().fillna(0).plot(ax=ax)
# Level: Easy / Hard

# Tests: groupby, Group barplot

killed_and_wounded_per_region = df.groupby('Region')[['Killed','Wounded']].sum()

killed_and_wounded_per_region.plot(kind='barh')



# or using seaborn

killed_and_wounded_per_region = df.groupby('Region')[['Killed','Wounded']].sum()

killed_and_wounded_per_region = killed_and_wounded_per_region.reset_index()

killed_and_wounded_per_region = killed_and_wounded_per_region.melt('Region',var_name='a', value_name='b')

sns.barplot(x='b', y='Region', hue='a', data=killed_and_wounded_per_region)

# Level: Easy

# Tests: Mode, Filtering

df[df['Country'] == 'India']['AttackType'].mode()
# Level: Hard

# Tests: Groupby, Working with series, mode

mode_attackType_per_country = df.groupby('AttackType')['Country'].apply(lambda x: x.mode())

mode_attackType_per_country
# Level: Easy

# Tests: Lineplot



fig, ax = plt.subplots(figsize=(15,7))

df[(df['AttackType'] == 'Armed Assault') & (df['Country'] == 'Pakistan')].groupby('Year').size().plot(ax=ax)
country_stats = pd.read_csv('/kaggle/input/undata-country-profiles/country_profile_variables.csv')
country_stats.head()
# Level: Hard

# Tests, Merge, Maipulating the dataset

merged = country_stats.merge(df,left_on='country',right_on='Country',suffixes=('_left', '_right'),how='right')

merged['Region'] = merged['Region_right']

merged['Population'] = merged['Population in thousands (2017)'] * 1000

merged= merged.groupby(['country'])[['Region','Population']].first().reset_index()



merged
# Level: Hard

# Tests: Scatterplot, working with series

countries_with_population = df[df['Country'].isin(merged['country'])]

attacks_per_citizen = countries_with_population['Country'].value_counts().sort_index() / merged.set_index('country')['Population']

killed_per_citizen = countries_with_population.groupby('Country')['Killed'].sum() / merged.set_index('country')['Population']

merged.set_index('country')['Region']

plot_df = pd.concat([attacks_per_citizen,killed_per_citizen,merged.set_index('country')['Region']], axis=1, keys=['attacks', 'killed','region'])





plt.subplots(figsize=(15,6))

sns.scatterplot(x='attacks', y='killed',data=plot_df, hue='region')