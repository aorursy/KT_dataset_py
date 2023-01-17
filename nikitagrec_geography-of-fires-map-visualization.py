from mpl_toolkits.basemap import Basemap

import matplotlib.pyplot as plt

from matplotlib.patches import Polygon

from matplotlib.collections import PatchCollection

from matplotlib.patches import PathPatch



import numpy as np

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.graph_objects as go
data = pd.read_csv('../input/forest-fires-in-brazil/amazon.csv',encoding="ISO-8859-1")

data.head()
fig = go.Figure()

for i in data['state'].unique():

    datas = data[data['state']==i][['date','state','number']].groupby(['date','state']).mean().reset_index()

    fig.add_trace(go.Scatter(x=datas['date'], y=datas['number'], name=i,

                        line_shape='linear'))

fig.show()
state_dict = {'Amapa':'Amapá', 'Ceara':'Ceará', 'Goias':'Goiás', 

 'Maranhao': 'Maranhão', 'Paraiba':'Paraíba', 

 'Piau':'Piauí', 'Rio':'Rio de Janeiro', 'Rondonia':'Rondônia', 'Sao Paulo':'São Paulo'}

data['state'].replace(state_dict, inplace=True)
from IPython.display import Image

Image('../input/brazil-img/brazil-states-map.gif',width=500, height=40)
cm = plt.get_cmap('afmhot')

plt.figure(figsize=(10,1))

for i in np.arange(250):

    plt.scatter(i,0, c=cm(i), s=1000);

    frame1 = plt.gca()

    frame1.axes.get_yaxis().set_visible(False)
def map_fires(year, ax):

    data_years = data[data['year']==year][['state','number']].groupby(['state']).mean().reset_index()

    cm = plt.get_cmap('afmhot')

    sns.set(style="white",font_scale=1.5)

#     fig = plt.figure(figsize=(10,10))

#     ax = fig.add_subplot(111)

    map = Basemap(projection='mill',llcrnrlat=-35,urcrnrlat=10,\

                llcrnrlon=-80,urcrnrlon=-30,resolution='c')

    map.drawparallels(np.arange(-90,91.,30.))

    map.drawmeridians(np.arange(-90,90.,60.))

    map.drawmapboundary(fill_color='aqua')

    map.fillcontinents(color='#ddaa66',lake_color='aqua', alpha=1)

    map.drawcoastlines()



    map.readshapefile('/kaggle/input/brazil-vers-3/gadm36_BRA_1', 'comarques', linewidth=2)



    for state in data_years['state'].unique():

        patches   = []

        for info, shape in zip(map.comarques_info, map.comarques):

            if info['NAME_1'] == state:

                patches.append( Polygon(np.array(shape), True) )

        ax.add_collection(PatchCollection(patches,facecolor=cm(int(data_years[data_years['state']==state]['number'].iloc[0])),

                                          linewidths=1., zorder=4)) 
fig = plt.figure(figsize=(40,40))

for i,year in enumerate([1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006]):

    ax1 = fig.add_subplot(3,3,i+1)

    ax1.set_title(year, fontsize=40)

    map_fires(year, ax1)

plt.tight_layout()
fig = plt.figure(figsize=(40,40))

# for i,year in enumerate(list(data['year'].unique())):

for i,year in enumerate([2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017]):

    ax1 = fig.add_subplot(3,3,i+1)

    ax1.set_title(year, fontsize=40)

    map_fires(year, ax1)

plt.tight_layout()
from PIL import Image

with Image.open('../input/brazil-img/Brazilian_states_by_population_2013.png') as img:

    width, height = (8,8)
from IPython.display import Image

plt.figure(figsize=(10,10));

Image('../input/brazil-img/Brazilian_states_by_population_2013.png',width=400, height=30)
plt.figure(figsize=(20,10))

sns.boxplot(data.year,data.number);