import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
# print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
from datetime import datetime
import pandas_datareader.data as web
# print(os.listdir('../input'))
dfAll = pd.read_csv('../input/openaq.csv')
dfAll['city'].unique()

#If you want to visualize some other city's pollution, then change Delhi below with one of available cities printed above
df = dfAll[dfAll['city'] == 'Delhi']
# df = df[df['location'] == 'Anand Vihar, Delhi - DPCC']
df = df[df.parameter == 'pm25']
df_pivot=df[['location', 'local', 'value']].pivot_table(index='location', aggfunc={np.mean})
df_pivot_sorted=df_pivot.reindex(df_pivot[df_pivot.columns[0]].sort_values(ascending=False).index)
df_pivot_sorted.head()
for i in range(0, 5):   #draw a plot of top 5 locations in selected city
    location_index = i
    data_to_analyase = df[df.location == df_pivot_sorted.index[location_index]] #take top contributor location
    data_to_analyase = data_to_analyase[data_to_analyase.value > 0]  #do not consider values with 0 or negative
    
    data_to_analyase_hourly = data_to_analyase
    data_to_analyase_hourly.index = pd.to_datetime(data_to_analyase['local'])
    data_to_analyase_hourly = data_to_analyase_hourly.resample('d').mean()

    layout = go.Layout(title=df_pivot_sorted.index[location_index])
    #safe level set as 60 : https://en.wikipedia.org/wiki/Air_pollution_in_Delhi
    data = [go.Scatter(x=data_to_analyase.local, y=data_to_analyase['value'], name='Actual PM2.5 level'), 
            go.Scatter(x=data_to_analyase.local, y=[60]*len(data_to_analyase), name='Safe Level (60)'),
            go.Scatter(x=data_to_analyase_hourly.index, y=data_to_analyase_hourly.value, name='Daily Average')
           ]
    fig = go.Figure(data, layout=layout)
    py.iplot(fig)

import matplotlib.pyplot as plt
import matplotlib.cm

from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize

import matplotlib.pyplot as plt
import matplotlib.cm
 
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize
from scipy.stats import gaussian_kde
#You can fork the notebook and beauify India map further.

# fig, ax = plt.subplots(figsize=(5,5))
fig, ax = plt.subplots(figsize=(10,10))
m = Basemap(
    llcrnrlon=68.0996943,llcrnrlat=5.610304,
#             #llcrnrlon=dfAll.longitude.min(), llcrnrlat=dfAll.latitude.min(),
            urcrnrlon=97.4,urcrnrlat=35.610304,
#             #urcrnrlon=dfAll.longitude.max(), urcrnrlat=dfAll.latitude.max(),
            resolution='i', # Set using letters, e.g. c is a crude drawing, f is a full detailed drawing
#             projection='merc', 
            ax = ax,
            lon_0=77.0996943,lat_0=28.610304 # Setting the central point of the image
           ) 
m.drawmapboundary(fill_color='#46bcec') # Make your map into any style you like
m.fillcontinents(color='#f2f2f2',lake_color='#46bcec') # Make your map into any style you like
# m.drawcoastlines()
# m.drawrivers() # Default colour is black but it can be customised
m.drawcountries()
m.drawstates()

dfplot = dfAll.dropna()[dfAll.value > 0]
dffinal = None
#Get a dataframe having day wise average pollution level
for c in dfplot.location.unique():
    data_to_analyase_daily = dfplot[dfplot.location == c]
    data_to_analyase_daily.index = pd.to_datetime(data_to_analyase_daily['local'])
#     data_to_analyase_daily = data_to_analyase_daily.resample('d').median()   #Day wise average
    data_to_analyase_daily = data_to_analyase_daily[data_to_analyase_daily.parameter == 'pm25']
#     print(data_to_analyase_daily.columns)
    df_pivot=data_to_analyase_daily[['location', 'local', 'latitude', 'longitude', 'value']].pivot_table(values=['latitude', 'longitude', 'value'], index='location', aggfunc={np.mean})
#     print(df_pivot.columns)
    if dffinal is not None:
        dffinal = pd.concat([dffinal, df_pivot])
    else:
        dffinal = df_pivot

# print(dffinal.columns)

safe_level = 60
marker_size = 4

cm = plt.cm.RdYlGn_r   #Green -> Yellow -> Red
ax.scatter(dffinal['longitude']['mean'], dffinal['latitude']['mean'], [x * marker_size if(x <= safe_level)  else safe_level * marker_size for x in dffinal['value']['mean'] if type(x) != str], marker='o', zorder=10, c=[x if(x <= safe_level)  else safe_level for x in dffinal['value']['mean'] if type(x) != str], cmap=cm)  
# cb = plt.colorbar(ax=ax)
# cb.set_label('Pollution Density')

plt.show()