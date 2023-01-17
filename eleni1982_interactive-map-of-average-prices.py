import numpy as np

import pandas as pd

import folium

import os

from IPython.display import display, HTML

train = pd.read_csv('../input/train.csv')
iowa_poly = os.path.join('../input/iowa_poly.json')
m = folium.Map(location = [42.0564052,-93.6442311], zoom_start = 7)
coords_v0 = pd.DataFrame(np.array(

        [

        ['Blmngtn','Bloomington Heights',42.0564052,-93.6442311]

       ,['Blueste','Bluestem',41.497925,-93.5011687]

       ,['BrDale','Briardale',42.5228147,-93.2860814]

       ,['BrkSide','Brookside',42.028952,-93.6319627]

       ,['ClearCr','Clear Creek',41.8199517,-93.3600346]

       ,['CollgCr','College Creek',42.0214541,-93.6671637]

       ,['Crawfor','Crawford',41.377671,-93.9140169]

       ,['Edwards','Edwards',42.0154064,-93.6875441]

       ,['Gilbert','Gilbert',42.1068336,-93.6553512]

       ,['IDOTRR','Iowa DOT and Rail Road',42.0220014,-93.6242068]

       ,['MeadowV','Meadow Village',42.0048434,-93.6568125]

       ,['Mitchel','Mitchell',43.3185572,-92.8779557]

       ,['NAmes','North Ames',42.059172,-93.6441717]

       ,['NoRidge','Northridge',42.0485012,-93.6526078]

       ,['NPkVill','Northpark Villa',42.0499088,-93.6290747]

       ,['NridgHt','Northridge Heights',42.0597767,-93.6500184]

       ,['NWAmes','Northwest Ames',42.042906,-93.6642637]

       ,['OldTown','Old Town',43.3135899,-95.1529172]

       ,['SWISU','South & West of Iowa State University',42.0318986,-93.6585304]

       ,['Sawyer','Sawyer',40.6964434,-91.3639064]

       ,['SawyerW','Sawyer West',40.706617,-91.3805747]

       ,['Somerst','Somerset',41.5233969,-93.6141585]

       ,['StoneBr','Stone Brook',42.059385,-93.6355362]

       ,['Timber','Timberland',41.720651,-91.4766478]

       ,['Veenker','Veenker',42.0416438,-93.6513107]

]), columns = ['Neighborhood','Neighborhood_FULL','Lat','Long']) 



coords_v0["Lat"] = pd.to_numeric(coords_v0["Lat"])

coords_v0["Long"] = pd.to_numeric(coords_v0["Long"])
mean_price_per_neighborhood = train.groupby('Neighborhood', as_index=False)['SalePrice'].mean()

mean_price_per_neighborhood['SalePrice'] = round(mean_price_per_neighborhood['SalePrice'])



coords = pd.merge(coords_v0, mean_price_per_neighborhood, on = 'Neighborhood')

display(HTML(coords.to_html()))
p_min = mean_price_per_neighborhood['SalePrice'].describe()['min']

p_50 = mean_price_per_neighborhood['SalePrice'].describe()['50%']

p_75 = mean_price_per_neighborhood['SalePrice'].describe()['75%']

p_max = mean_price_per_neighborhood['SalePrice'].describe()['max']
def get_color(x):        

    if (p_min <= x < p_50):

        return "../input/green.png"

    elif(p_50 <= x <= p_75):

        return "../input/blue.png"

    elif(p_75 < x <= p_max):

        return "../input/red.png" 
for index, row in coords.iterrows():

    icon_path = get_color(row['SalePrice'])

    logo_icon = folium.features.CustomIcon(icon_image=icon_path ,icon_size=(18,18))

    folium.Marker([row['Lat'],row['Long']]

    ,popup = row['Neighborhood_FULL']+'\n AVG_Price:' +str(row['SalePrice'])

    ,icon=logo_icon).add_to(m)
folium.GeoJson(iowa_poly, name = 'iowa_poly').add_to(m)
m