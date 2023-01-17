import numpy as np

import pandas as pd

import seaborn as sns

import random

import plotly.graph_objects as go

import plotly.express as px

import scipy.stats as stt

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')

%pylab inline
data15 = pd.read_csv('../input/world-happiness/2015.csv')

data15.head()
fig = plt.figure(figsize=(20,20))

sns.set(style="white",font_scale=1);

sns.pairplot(data15[['Happiness Score','Economy (GDP per Capita)', 'Family', 'Health (Life Expectancy)', \

    'Freedom', 'Trust (Government Corruption)']]);
fig = plt.figure(figsize=(13,10))

sns.set(style="white",font_scale=1.5)

sns.heatmap(data15.dropna()[['Happiness Score','Economy (GDP per Capita)', 'Family', 'Health (Life Expectancy)', \

    'Freedom', 'Trust (Government Corruption)','Generosity', 'Dystopia Residual']].corr(), fmt='.2f',annot=True,\

             xticklabels=False,linewidth=2);
vis15 = data15[['Happiness Score','Economy (GDP per Capita)','Family','Country']]



fig = go.Figure()



for x in range(40):

    fig.add_trace(go.Scatter(

        x=[vis15.iloc[x,:]['Happiness Score'], vis15.iloc[x,:]['Happiness Score']],

        y=[vis15.iloc[x,:]['Economy (GDP per Capita)'],vis15.iloc[x,:]['Family']],

        mode="markers+lines+text",

        text = [vis15.iloc[x,:]['Country']],

        textposition="bottom center"

    ))



    fig.update_layout(  

        showlegend=False,

    )

fig.show()
fig = px.scatter(data15, x='Happiness Score', y='Economy (GDP per Capita)', text='Country')

fig.update_traces(textposition='top center')

fig.update_layout(

    height=1000,

)

fig.show()
fig = plt.figure(figsize=(7,5))

sns.set()

sns.distplot(data15['Happiness Score'],bins=12);
from mpl_toolkits.basemap import Basemap
concap = pd.read_csv('../input/world-capitals-gps/concap.csv')

concap.head()
data_full = pd.merge(concap[['CountryName', 'CapitalName', 'CapitalLatitude', 'CapitalLongitude']],\

         data15,left_on='CountryName',right_on='Country')
def mapWorld():

    m = Basemap(projection='mill',llcrnrlat=-60,urcrnrlat=90,\

            llcrnrlon=-180,urcrnrlon=180,resolution='c')

    #m.drawcoastlines()

    m.drawcountries()

    m.drawparallels(np.arange(-90,91.,30.))

    m.drawmeridians(np.arange(-90,90.,60.))

    

    #m.drawmapboundary(fill_color='#FFFFFF')

    lat = data_full['CapitalLatitude'].values

    lon = data_full['CapitalLongitude'].values

    a_1 = data_full['Happiness Rank'].values

    a_2 = data_full['Economy (GDP per Capita)'].values

    m.scatter(lon, lat, latlon=True,c=100*a_1,s=1000*a_2,linewidth=1,edgecolors='black',cmap='hot', alpha=1)

    

    m.fillcontinents(color='#072B57',lake_color='#FFFFFF',alpha=0.4)

    cbar = m.colorbar()

    cbar.set_label('Happiness Rank*1000')

    #plt.clim(20000, 100000)

    plt.title("World Happiness Rank", fontsize=30)

    plt.show()

sns.set(style="white",font_scale=1.5)

plt.figure(figsize=(30,30))

mapWorld()
data_full['Economy (GDP per Capita) color'] = data_full['Economy (GDP per Capita)'].apply(lambda x: int(100*x))

data_full['Happiness Rank size'] = data_full['Happiness Rank'].apply(lambda x: int(x//5))

fig = go.Figure()



fig.add_trace(go.Scattergeo(

                         lat = data_full['CapitalLatitude'],

                     lon = data_full['CapitalLongitude'],

                     text = data_full['CountryName'],

#                     mode="markers+text",|

                     marker={

                        "color": data_full['Economy (GDP per Capita) color'],

                        "line": {"width": 1},

                 "size": data_full['Happiness Rank size']}

                           ))



fig.update_layout(

    title_text="World Happiness Rank"

)



fig.show()
contr_list = list(data15[data15['Region'].isin(['Western Europe','Central and Eastern Europe'])]['Country'].unique())

eu_gps = concap[concap['CountryName'].isin(contr_list)]

eu_data = data15[data15['Region'].isin(['Western Europe','Central and Eastern Europe'])]

eu_full = pd.merge(eu_gps[['CountryName', 'CapitalName', 'CapitalLatitude', 'CapitalLongitude']],\

         eu_data,left_on='CountryName',right_on='Country')
def mapEurope(column_color, column_size,colbar=True):

    m = Basemap(projection='mill',llcrnrlat=30,urcrnrlat=72,\

                llcrnrlon=-20,urcrnrlon=55,resolution='l')

    m.drawcountries()

    m.drawstates()

    #m.drawmapboundary()

    m.drawparallels(np.arange(-90,91.,30.))

    m.drawmeridians(np.arange(-90,90.,60.))

    lat = eu_full['CapitalLatitude'].values

    lon = eu_full['CapitalLongitude'].values

    a_1 = eu_full[column_color].values

    a_2 = eu_full[column_size].values

    #s=1000*a_2

    m.scatter(lon, lat, latlon=True,c=1000*a_1,s=1000*a_2,linewidth=2,edgecolors='black',cmap='hot', alpha=1)

    m.fillcontinents(color='#072B57',lake_color='#FFFFFF',alpha=0.3)

    if colbar:

            m.colorbar(label='Happiness Rank*1000')

    else:pass

plt.figure(figsize=(15,15))

plt.title('Europe - Happiness\GDP', fontsize=30)

mapEurope('Happiness Rank','Economy (GDP per Capita)')
fig = plt.figure(figsize=(30,15))

ax1 = fig.add_subplot(2,3,1)

ax1.set_title('Europe - Happiness\Freedom', fontsize=20)

mapEurope('Happiness Rank','Freedom',colbar=False)

ax2 = fig.add_subplot(2,3,2)

ax2.set_title('Europe - Happiness\Government Corruption', fontsize=20)

mapEurope('Happiness Rank','Trust (Government Corruption)')

plt.tight_layout()
plt.figure(figsize=(8,8))

plt.title('Europe - Happiness\Family', fontsize=20)

mapEurope('Happiness Rank','Family')