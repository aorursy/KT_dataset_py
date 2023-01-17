import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # this is used for the plot the graph 

import seaborn as sns # used for plot interactive graph.



import matplotlib.pyplot as plt

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go



import warnings

warnings.filterwarnings('ignore')

from pylab import rcParams

# figure size in inches



%matplotlib inline

import warnings

warnings.filterwarnings('ignore')

import os

print(os.listdir("../input"))

dataset = pd.read_csv('../input/suicide-rates-overview-1985-to-2016/master.csv')

dataset.head()
unique_country = dataset['country'].unique()

print(unique_country)
x = dataset['sex'].dropna()

y = dataset['age'].dropna()

z = dataset['suicides_no'][dataset.suicides_no!=0].dropna()

p = dataset['population'][dataset.population!=0].dropna()

t = dataset['suicides/100k pop'].dropna()

gdp = dataset['gdp_per_capita ($)']



p = sns.pairplot(pd.DataFrame(list(zip(x, y, np.log(z), np.log10(p), t, gdp)), 

                        columns=['sex','age', 'suicides_no', 'population', 'suicides/100k pop', 'gdp_per-capita ($)']), hue='sex', palette="Set2")
import plotly

# connected=True means it will download the latest version of plotly javascript library.

plotly.offline.init_notebook_mode(connected=True)

import plotly.graph_objs as go



generation = dataset['generation'].value_counts().sort_values(ascending=True)



data = [go.Pie(

        labels = generation.index,

        values = generation.values,

        hoverinfo = 'label+value')]



plotly.offline.iplot(data, filename='active_generation')
groups = dataset.groupby('country').filter(lambda x: len(x) >= 50).reset_index()

sns.set_style("darkgrid")

ax = sns.jointplot(dataset['suicides_no'], dataset['population'])
c = ['hsl('+str(h)+',50%'+',50%)' for h in np.linspace(0, 360, len(list(set(groups.country))))]

subset_dataset = dataset[dataset.population> 50000]

groups_temp = subset_dataset.groupby('country').filter(lambda x: len(x) >20)



for country in enumerate(list(set(groups_temp.country))):



    data = [{

    'x': groups_temp.loc[subset_dataset.country==country[1]]['suicides_no'],

    'type':'scatter',

    'y' : subset_dataset['population'],

    'name' : str(country[1]),

    'mode' : 'markers',

    'showlegend': True,

   

    } for country in enumerate(['Colombia' ,'Suriname'])]



layout = {'title':"suicides_no vs population", 

          'xaxis': {'title' : 'suicides_no'},

          'yaxis' : {'title' : 'population'},

         'plot_bgcolor': 'rgb(0,0,0)'}



plotly.offline.iplot({'data': data, 'layout': layout})

data = [{

    'x': groups_temp.loc[subset_dataset.sex==sex[1]]['suicides_no'],

    'type':'scatter',

    'y' : subset_dataset['population'],

    'name' : str(sex[1]),

    'mode' : 'markers',

    'showlegend': True,

   

    } for sex in enumerate(['male' ,'female'])]



layout = {'title':"suicides_no vs population", 

          'xaxis': {'title' : 'suicides_no'},

          'yaxis' : {'title' : 'population'},

         'plot_bgcolor': 'rgb(0,0,0)'}



plotly.offline.iplot({'data': data, 'layout': layout})

subset_df = dataset[dataset.generation.isin(['Boomers', 'Millenials', 'Silent', 'Generation X', 'Generation Z', 'G.I. Generation'])]

sns.set_style('darkgrid')

fig, ax = plt.subplots()

fig.set_size_inches(15, 8)

p = sns.stripplot(x="suicides/100k pop", y="suicides_no", data=subset_df, jitter=True, linewidth=1)

title = ax.set_title('')
by_year = dataset['suicides_no'].groupby([dataset['year'], dataset['sex']]).agg({'suicides_no':sum}).assign(percent = lambda x: 100 * x/x.sum())

by_year = np.round(by_year, decimals=2)

by_year = by_year.reset_index().sort_values(by='suicides_no', ascending=False)

most_year = by_year

print(most_year.head(5))





fig = plt.figure(figsize=(16,4))

ax = sns.lineplot(x="year", y="suicides_no",

hue="sex", style="sex",

markers=True, dashes=False, data=most_year)
suic_sex = dataset['suicides_no'].groupby(dataset['sex']).agg({'suicides_no' : 'sum'}).assign(percent = lambda x: 100 * x/x.sum())

suic_sex = np.round(suic_sex, decimals=0)

suic_sex = suic_sex.reset_index().sort_values(by='suicides_no',ascending=False)

most_sex = suic_sex

print("Total and percent of suicides among genders in year 1985 - 2016")

print()

print(most_sex)

fig = plt.figure(figsize=(6,4))

plt.title('Suicides by sex.')

sns.set(font_scale=0.9)

sns.barplot(y='suicides_no',x='sex',data=most_sex,palette="OrRd");

plt.ylabel('Number of suicides')

plt.tight_layout()
import pandas as pd

concap = pd.read_csv('../input/world-capitals-gps/concap.csv')

print(concap.head())
from mpl_toolkits.basemap import Basemap

by_country = dataset['suicides_no'].groupby([dataset['country']]).agg({'suicides_no':'sum'}).assign(percent = lambda x: 100 * x/x.sum())

by_country = np.round(by_country, decimals=0)

by_country = by_country.reset_index().sort_values(by='suicides_no', ascending=False)

most_country = by_country.head(15)

print("Total and percent of suicides among countries in year 1985 - 2016")

print()

print(most_country.head(15))

fig = plt.figure(figsize=(20,6))

plt.title('Suicides by country')

sns.set(font_scale=0.9)

sns.barplot(y='suicides_no',x='country',data=most_country,palette='Set2',color='');

plt.xlabel('Countries')

plt.ylabel('suicides_no')

plt.tight_layout()



data_full = pd.merge(concap[['CountryName', 'CapitalName', 'CapitalLatitude', 'CapitalLongitude']],\

         by_country,left_on='CountryName',right_on='country')

def mapWorld(col1,size2,title3,label4,metr=100,colmap='hot'):

    m = Basemap(projection='mill',llcrnrlat=-60,urcrnrlat=70,\

            llcrnrlon=-110,urcrnrlon=180,resolution='c')

    m.drawcoastlines()

    m.drawcountries()

    m.drawparallels(np.arange(-90,91.,30.))

    m.drawmeridians(np.arange(-90,90.,60.))

    



    lat = data_full['CapitalLatitude'].values

    lon = data_full['CapitalLongitude'].values

    a_1 = data_full[col1].values

    if size2:

        a_2 = data_full[size2].values

    else: a_2 = 1



    m.scatter(lon, lat, latlon=True,c=a_1,s=metr*a_2,linewidth=1,edgecolors='black',cmap=colmap, alpha=1)

    

    cbar = m.colorbar()

    cbar.set_label(label4,fontsize=30)

    plt.title(title3, fontsize=30)

    plt.show()

sns.set(font_scale=1.5)

plt.figure(figsize=(20,10))

mapWorld(col1='suicides_no', size2=False,title3='Suicides by countries',label4='',metr=150,colmap='cool')