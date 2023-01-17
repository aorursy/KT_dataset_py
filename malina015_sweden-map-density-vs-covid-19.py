import cartopy.crs as ccrs

import cartopy.io.shapereader as shpreader

import matplotlib.pyplot as plt

import geopandas as gpd

import geoplot

import pandas as pd
fname = '../input/gadm36_SWE_shp/gadm36_SWE_1.shp'

se_shapes = list(shpreader.Reader(fname).geometries())

ax = plt.axes(projection=ccrs.PlateCarree())

plt.title('Sweden')

ax.coastlines(resolution='10m')

ax.add_geometries(se_shapes, ccrs.PlateCarree(),

                  edgecolor='black', facecolor='gray', alpha=0.5)

ax.set_extent([11, 25, 55, 70], ccrs.PlateCarree())

plt.show()
map_df = gpd.read_file(fname)

map_df.crs
data_proj = map_df.copy()

data_proj = data_proj.to_crs(epsg=3035)
import matplotlib.pyplot as plt



data_proj.plot(facecolor='green');

plt.title("New projection");

plt.tight_layout()

sweden_pop = pd.read_excel('../input/be0101_tabkv1_2020eng_2.xlsx')
sweden_pop = sweden_pop.drop([0,312,313,314,315],axis=0)

sweden_pop['County'] = sweden_pop['County'].str.rstrip()

sweden_pop['County'] = sweden_pop['County'].apply(lambda row: 'Orebro' if row=='Örebro' else row)

sweden_pop.head()
sweden_pop['is_county'] = sweden_pop.apply(lambda row: 1 if len(row['Code'])==2 else 0, axis = 1)

sweden_pop_by_county = sweden_pop[sweden_pop['is_county']==1]

sweden_map_pop = pd.merge(data_proj, sweden_pop_by_county,how = 'outer', left_on = 'NAME_1', right_on = 'County')
import mapclassify

scheme = mapclassify.Quantiles(sweden_map_pop['Population'], k=6)

geoplot.choropleth(

    sweden_map_pop, hue=sweden_map_pop['Population'], scheme=scheme,

    cmap='Blues', figsize=(4, 12)

)
import matplotlib.colors as mplc



fig, ax = plt.subplots(1, figsize=(30, 10))

ax.axis('off')

ax.set_title('Popoulation by counties in Sweden', fontdict={'fontsize': '18', 'fontweight' : '3'})

sm = plt.cm.ScalarMappable(cmap='Blues',

                           norm=plt.Normalize()

                          )

fig.colorbar(sm)

sweden_map_pop.plot(column='Population', cmap='Blues', linewidth=0.8, ax=ax, edgecolor='0.8',norm=mplc.LogNorm())

from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np

vmin=min(sweden_map_pop['Population'])

vmax=max(sweden_map_pop['Population'])

fig, ax = plt.subplots(1, figsize=(30, 10))

ax.axis('off')

divider = make_axes_locatable(ax)

cax = divider.append_axes("right", size="5%", pad=0.1)

base = sweden_map_pop.plot(column='Population',cmap='viridis', ax=ax, legend=True,

                     linewidth=0.8, norm=mplc.LogNorm(vmin, vmax),

                             edgecolor='0.8'

                      , cax=cax

                     )

colourbar = ax.get_figure().get_axes()[1]

yticks = np.interp(colourbar.get_yticks(), [0,1], [vmin, vmax])

colourbar.set_yticklabels(['{0:.0f}'.format(ytick) for ytick in yticks])

plt.show()
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np

vmin=min(sweden_map_pop['Population'])

vmax=max(sweden_map_pop['Population'])

fig, ax = plt.subplots(1, figsize=(30, 10))

ax.axis('off')

divider = make_axes_locatable(ax)

base = sweden_map_pop.plot(column='Population',cmap='Reds', ax=ax, legend=True,

                     linewidth=0.8, 

                             edgecolor='0.8', scheme='maxp'

                     )

plt.show()
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np

vmin=min(sweden_map_pop['Population'])

vmax=max(sweden_map_pop['Population'])

fig, ax = plt.subplots(1, figsize=(30, 10))

ax.axis('off')

divider = make_axes_locatable(ax)

base = sweden_map_pop.plot(column='Population',cmap='Reds', ax=ax, legend=True,

                     linewidth=0.8, 

                             edgecolor='0.8', scheme='maxp'

                     )

leg = ax.get_legend()

leg.set_bbox_to_anchor((0., 0.8, 0.3, 0.1))

for lbl in leg.get_texts():

    label_text = lbl.get_text()

    lower = label_text.split()[0]

    upper = label_text.split()[2]

    new_text = f'{float(lower):,.0f} - {float(upper):,.0f}'

    lbl.set_text(new_text)

plt.show()
import json

import pandas as pd

import urllib
import requests

url = 'https://api.apify.com/v2/datasets/Nq3XwHX262iDwsFJS/items?format=json&clean=1'

r = requests.get(url)

print(r.json()[0:2]) 
covid_19_data_json = r.json()[len(r.json())-1]
covid_19_data = pd.DataFrame(covid_19_data_json['infectedByRegion'], index=list(range(0,21,1)))

covid_19_data.head()


covid_19_data['region'] = covid_19_data['region'].apply(lambda row: 'Orebro' if row=='Örebro' else row)

covid_19_data['region'] = covid_19_data['region'].apply(lambda row: 'Södermanland' if row=='Sörmland' else row)

covid_19_data['region'] = covid_19_data['region'].apply(lambda row: 'Jämtland' if row=='Jämtland Härjedalen' else row)

sweden_map_covid = pd.merge(sweden_map_pop, covid_19_data,how = 'outer', left_on = 'NAME_1', right_on = 'region') 

sweden_map_covid.head()
from mpl_toolkits.axes_grid1 import make_axes_locatable

import matplotlib.cm as cm

import numpy as np

vmin=min(sweden_map_covid['infectedCount'])

vmax=max(sweden_map_covid['infectedCount'])

fig, ax = plt.subplots(1, figsize=(30, 10))

ax.axis('off')

divider = make_axes_locatable(ax)

cax = divider.append_axes("right", size="5%", pad=0.1)

cmap_reversed = cm.get_cmap('viridis_r')

base = sweden_map_covid.plot(column='infectedCount',cmap=cmap_reversed, ax=ax, legend=True,

                     linewidth=0.8, norm=mplc.LogNorm(vmin, vmax),

                             edgecolor='0.8'

                      , cax=cax

                     )

colourbar = ax.get_figure().get_axes()[1]

yticks = np.interp(colourbar.get_yticks(), [0,1], [vmin, vmax])

colourbar.set_yticklabels(['{0:.0f}'.format(ytick) for ytick in yticks])

fig.savefig('Map_of_infected.png', dpi=150, transparent=True)

plt.show()