# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import plotly as py

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go



import warnings

warnings.filterwarnings('ignore')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

print(os.listdir("../input/"))

# Any results you write to the current directory are saved as output.
world_original=pd.read_csv('/kaggle/input/countries-of-the-world/countries of the world.csv')
world_original.head(10)
world_original.sample(10)
world_original.shape
world_original.info()
world_original.isnull().sum()
world_original['Country'].nunique()
world_original['Climate'].unique()
world_original['Region'].unique()
world_original.describe()
world_original.columns
world=world_original.copy()
world['Country']=world['Country'].str.strip()

world['Region']=world['Region'].str.strip()
world[['Country','Region']]
world.replace(',', '.', regex=True,inplace=True)
world
world['Region'].replace(['ASIA (EX. NEAR EAST)', 'NEAR EAST'], ['EASTERN ASIA','WESTERN ASIA'], regex=False,inplace=True)
list(world['Region'].unique())
world[['Population','Pop. Density (per sq. mi.)','Coastline (coast/area ratio)','Net migration','Infant mortality (per 1000 births)','GDP ($ per capita)','Literacy (%)','Phones (per 1000)','Arable (%)','Crops (%)','Other (%)','Birthrate','Deathrate','Agriculture','Industry','Service']]= world[['Population','Pop. Density (per sq. mi.)','Coastline (coast/area ratio)','Net migration','Infant mortality (per 1000 births)','GDP ($ per capita)','Literacy (%)','Phones (per 1000)','Arable (%)','Crops (%)','Other (%)','Birthrate','Deathrate','Agriculture','Industry','Service']].apply(pd.to_numeric, errors='coerce')

world['Climate']=world['Climate'].astype('category')

world['Region']=world['Region'].astype('category')
world.info()
world['Climate'].unique()

world['Climate'].replace(['1.5', '2.5'], ['5', '6'], regex=False,inplace=True)
world['Climate'].unique()
world[world['Net migration'].isnull()]
world['Net migration'].loc[[47,221]]=-21.1

world['Net migration'].loc[[223]]=-4.9
world[world['Infant mortality (per 1000 births)'].isnull()]
world['Infant mortality (per 1000 births)'].loc[[47,221]]=12.6

world['Infant mortality (per 1000 births)'].loc[[223]]=50.5
world[world['GDP ($ per capita)'].isnull()]
world['GDP ($ per capita)'].loc[[223]]=2500.0
world[(world['Phones (per 1000)'].isnull()) | (world['Arable (%)'].isnull()) | (world['Crops (%)'].isnull()) | (world['Other (%)'].isnull())]
world['Phones (per 1000)'].loc[[52,58,140,223]]=[330.56,2.48,882.2,256.4]

world['Literacy (%)']=world.groupby("Region").apply(lambda x: x.fillna(x.mean()))['Literacy (%)']

world['Arable (%)']=world.groupby("Region").apply(lambda x: x.fillna(x.mean()))['Arable (%)']

world['Crops (%)']=world.groupby("Region").apply(lambda x: x.fillna(x.mean()))['Crops (%)']

world['Other (%)']=world.groupby("Region").apply(lambda x: x.fillna(x.mean()))['Other (%)']

world['Birthrate']=world.groupby("Region").apply(lambda x: x.fillna(x.mean()))['Birthrate']

world['Deathrate']=world.groupby("Region").apply(lambda x: x.fillna(x.mean()))['Deathrate']

world['Agriculture']=world.groupby("Region").apply(lambda x: x.fillna(x.mean()))['Agriculture']

world['Industry']=world.groupby("Region").apply(lambda x: x.fillna(x.mean()))['Industry']

world['Service']=world.groupby("Region").apply(lambda x: x.fillna(x.mean()))['Service']

world['Climate']=world.groupby("Region").apply(lambda x: x.fillna(world['Climate'].mode()[0]))['Climate']
world.isnull().sum()
world.rename(columns={"Coastline (coast/area ratio)":"Coastline/Area (coast/area ratio)",'Net migration':'Net migration (per year)', 'Birthrate':'Birthrate (per 1000)', 'Deathrate':'Deathrate (per 1000)', 'Agriculture':'Agriculture (%)', 'Industry':'Industry (%)', 'Service':'Service (%)'},inplace=True)
world.info()
world.info()
world.head(10)
world.sample(10)
plt.style.use("fivethirtyeight")
plt.subplots(figsize=(30,15))

sns.countplot(world['Region'],order=world['Region'].value_counts().index)

plt.show()
world['Population'].describe()
plt.subplots(figsize=(15,10))

sns.distplot(world['Population'])

plt.show()
world['Region'].value_counts().index
world['Area (sq. mi.)'].describe()
plt.subplots(figsize=(15,10))

sns.distplot(world['Area (sq. mi.)'])

plt.show()
world['Pop. Density (per sq. mi.)'].describe()
plt.subplots(figsize=(15,10))

sns.distplot(world['Pop. Density (per sq. mi.)'])

plt.show()
world['Coastline/Area (coast/area ratio)'].describe()
plt.subplots(figsize=(15,10))

sns.distplot(world['Coastline/Area (coast/area ratio)'])

plt.show()
world['Net migration (per year)'].describe()
plt.subplots(figsize=(15,10))

sns.distplot(world['Net migration (per year)'])

plt.show()
world['Infant mortality (per 1000 births)'].describe()
plt.subplots(figsize=(15,10))

sns.distplot(world['Infant mortality (per 1000 births)'])

plt.show()
world['GDP ($ per capita)'].describe()
world[world['GDP ($ per capita)']>world['GDP ($ per capita)'].std()].sort_values(by='GDP ($ per capita)',ascending=False)[['Country','GDP ($ per capita)']].iloc[:5]
world[world['GDP ($ per capita)']<world['GDP ($ per capita)'].std()].sort_values(by='GDP ($ per capita)')[['Country','GDP ($ per capita)']].iloc[:5]
plt.subplots(figsize=(15,10))

sns.distplot(world['GDP ($ per capita)'])

plt.show()
world['Literacy (%)'].describe()
plt.subplots(figsize=(15,10))

sns.distplot(world['Literacy (%)'])

plt.show()
world['Phones (per 1000)'].describe()
plt.subplots(figsize=(15,10))

sns.distplot(world['Phones (per 1000)'])

plt.show()
world['Arable (%)'].describe()
plt.subplots(figsize=(15,10))

sns.distplot(world['Arable (%)'])

plt.show()
world['Crops (%)'].describe()
plt.subplots(figsize=(15,10))

sns.distplot(world['Crops (%)'])

plt.show()
world['Other (%)'].describe()
plt.subplots(figsize=(15,10))

sns.distplot(world['Other (%)'])

plt.show()
world['Climate'].describe()
plt.subplots(figsize=(15,10))

sns.countplot(x=world['Climate'],order=world['Climate'].value_counts().index)

plt.show()
world['Birthrate (per 1000)'].describe()
plt.subplots(figsize=(15,10))

sns.distplot(world['Birthrate (per 1000)'])

plt.show()
world['Deathrate (per 1000)'].describe()
plt.subplots(figsize=(15,10))

sns.distplot(world['Deathrate (per 1000)'])

plt.show()
world['Agriculture (%)'].describe()
plt.subplots(figsize=(15,10))

sns.distplot(world['Agriculture (%)'])

plt.show()
world['Industry (%)'].describe()
plt.subplots(figsize=(15,10))

sns.distplot(world['Industry (%)'])

plt.show()
world['Service (%)'].describe()
plt.subplots(figsize=(15,10))

sns.distplot(world['Service (%)'])

plt.show()
plt.figure(figsize=(20,20))

sns.heatmap(world.corr(),annot=True)

plt.show()


x = world.loc[:,["Region","GDP ($ per capita)","Population","Infant mortality (per 1000 births)","Literacy (%)",'Birthrate (per 1000)']]

sns.pairplot(x,hue="Region",height=8,aspect=.5)

plt.show()
sns.catplot(x="Region", y="Population", kind="bar", data=world.groupby('Region').sum().reset_index(), height=10,aspect=2.5);

sns.catplot(x="Country", y="Population", kind="bar", data=world.nlargest(20, 'Population'), height=10,aspect=2.8);
#Population per country

data = dict(type='choropleth',

locations = world.Country,

locationmode = 'country names', z = world.Population,

text = world.Country, colorbar = {'title':'Population'},

colorscale = 'Blackbody', reversescale = True)

layout = dict(title='Population per country',

geo = dict(showframe=False,projection={'type':'natural earth'}))

choromap = go.Figure(data = [data],layout = layout)

iplot(choromap,validate=False)
data = dict(type='choropleth',

locations = world.Country,

locationmode = 'country names', z = world['Infant mortality (per 1000 births)'],

text = world.Country, colorbar = {'title':'Infant Mortality'},

colorscale = 'YlOrRd', reversescale = False)

layout = dict(title='Infant Mortality per Country',

geo = dict(showframe=False,projection={'type':'natural earth'}))

choromap = go.Figure(data = [data],layout = layout)

iplot(choromap,validate=False)
#Population per country

data = dict(type='choropleth',

locations = world.Country,

locationmode = 'country names', z = world['GDP ($ per capita)'],

text = world.Country, colorbar = {'title':'GDP'},

colorscale = 'Hot', reversescale = True)

layout = dict(title='GDP of World Countries',

geo = dict(showframe=False,projection={'type':'natural earth'}))

choromap = go.Figure(data = [data],layout = layout)

iplot(choromap,validate=False)
sns.catplot(x="Region", y="Pop. Density (per sq. mi.)", kind="bar", data=world.groupby('Region').sum().reset_index(), height=10,aspect=2.5);

sns.catplot(x="Country", y="Pop. Density (per sq. mi.)", kind="bar", data=world.nlargest(20, 'Pop. Density (per sq. mi.)'), height=10,aspect=2.8);
sns.relplot(x="Literacy (%)", y="GDP ($ per capita)", data=world, hue='Region', height=12);
sns.relplot(x="Literacy (%)", y="Infant mortality (per 1000 births)", data=world, hue='Region', height=12);
sns.relplot(x="Birthrate (per 1000)", y="Infant mortality (per 1000 births)", data=world, hue='Region', height=12);

sns.relplot(x="Deathrate (per 1000)", y="Infant mortality (per 1000 births)", data=world, hue='Region', height=12);
plt.figure(figsize=(15,15))

plt.pie((world['Birthrate (per 1000)']>world['Deathrate (per 1000)']).value_counts(),labels=['Birthrate>Deathrate','Deathrate>Birthrate'],autopct='%1.2f%%',shadow=True,startangle=90,explode=[0.15,0]);
sns.catplot(x="Climate", y="Population", kind="bar", data=world.groupby('Climate').sum().nlargest(20, 'GDP ($ per capita)').reset_index(), height=10,aspect=2.5);
sns.relplot(x="Agriculture (%)", y="GDP ($ per capita)", data=world, hue='Region', height=12);

sns.relplot(x="Industry (%)", y="GDP ($ per capita)", data=world, hue='Region', height=12);

sns.relplot(x="Service (%)", y="GDP ($ per capita)", data=world, hue='Region', height=12);