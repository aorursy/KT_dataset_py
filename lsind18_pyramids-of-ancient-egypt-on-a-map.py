import pandas as pd

import numpy as np

import folium

import matplotlib.pyplot as plt # plotting data

import seaborn as sns 

sns.set(color_codes=True)
prmds = pd.read_csv('/kaggle/input/egyptianpyramids/pyramids.csv')

prmds
prmds.info()
prmds.describe()
ax = sns.scatterplot(x='Base1 (m)',y='Base2 (m)', data=prmds)
prmds[prmds['Base1 (m)'] != prmds['Base2 (m)']]
lat_mean = prmds['Latitude'].mean()

lon_mean = prmds['Longitude'].mean()



periods = {'EDP': (1,2), 'Old': (3,4,5,6,7), 'FIP': (8,9,10), 'Middle': (11,12,13,14), 'SIP': (15,16,17), 'New': (18,)}

colors_dyn = {'EDP':'violet', 'Old': 'green', 'FIP': 'blue', 'Middle': 'orange', 'SIP': 'yellow', 'New': 'red'}



def col_dyn(dyn):

    for k, v in periods.items():

        if dyn in v:

            c = colors_dyn[k]

            break

        else:

            c = 'black'

    return c



legend_kingdom = '''

     <div style= "position: fixed; 

     bottom: 45px; left: 15px; width: 150px; height: 150px;white-space: pre-line; 

     border:1px solid grey; z-index:9999; font-size:10px; background-color:white;

     ">&nbsp; <b>Period</b>

    &nbsp; Early Dyn. Period &emsp;<i class="fa fa-caret-up fa-2x"

                  style="color:violet"></i>

     &nbsp; Old Kingdom &emsp;<i class="fa fa-caret-up fa-2x"

                  style="color:green"></i>

     &nbsp; FIP &emsp;<i class="fa fa-caret-up fa-2x"

                  style="color:lightblue"></i>

     &nbsp; Middle Kingdom &emsp;<i class="fa fa-caret-up fa-2x"

                  style="color:orange"></i>

    &nbsp; SIP &emsp;<i class="fa fa-caret-up fa-2x"

                  style="color:yellow"></i>

    &nbsp; New Kingdom &emsp;<i class="fa fa-caret-up fa-2x"

                  style="color:red"></i>

      </div>

     '''
m = folium.Map(location = [lat_mean, lon_mean], #center of a map

               zoom_start=6, min_zoom = 5, control_scale = True, tiles='Stamen Terrain')



for i in range(0,len(prmds)):

    lat = prmds.iloc[i]['Latitude']

    lon = prmds.iloc[i]['Longitude']

    folium.Marker([lat, lon], popup=prmds.iloc[i]['Modern name'], # show a modern name of pyramid in popup

                  icon=folium.Icon(color=col_dyn(prmds.iloc[i]['Dynasty']),

                                   icon='eye-close' if str(prmds.iloc[i]['Pharaoh']).find("?")==-1 else 'eye-open') # glyphicon with open eye if Pharaoh is certain

                 ).add_to(m)

m.get_root().html.add_child(folium.Element(legend_kingdom))

display(m)
prmds_Le = prmds.dropna(subset=['Lepsius'])

m2 = folium.Map(location = [prmds_Le['Latitude'].mean(), prmds_Le['Longitude'].mean()],

               zoom_start=9, min_zoom = 6, tiles='Stamen Toner')



for i in range(0,len(prmds_Le)):

    lat = prmds_Le.iloc[i]['Latitude']

    lon = prmds_Le.iloc[i]['Longitude']

    temp = prmds_Le.iloc[i]['Lepsius']

    folium.CircleMarker([lat, lon],  radius=8,  color='red', fill=True, fill_color='red').add_to(m2)

    folium.Marker([lat, lon], 

                  popup=prmds_Le.iloc[i]['Pharaoh'], 

                         icon=folium.DivIcon(html=f"""<div style="font-family: arial; color: 'black'">{"{}".format(temp)}</div>""")

                

        ).add_to(m2)

display(m2)