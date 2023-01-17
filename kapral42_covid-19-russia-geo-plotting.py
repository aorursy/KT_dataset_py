import pandas as pd

import geopandas as gpd

import numpy as np



import folium

from folium import Marker

from folium.plugins import HeatMap

from branca.colormap import linear



import geopandas as gpd

from geopandas.tools import geocode
rus_df = pd.read_csv('../input/covid19-russia-regions-cases/covid19-russia-cases.csv')

rus_other_df = pd.read_csv('../input/covid19-russia-regions-cases/covid19-tests-and-other.csv')

rus_info_df = pd.read_csv('../input/covid19-russia-regions-cases/regions-info.csv')

rus_vent_df = pd.read_csv('../input/covid19-russia-regions-cases/regions-ventilators.csv')

mos_addr_df = pd.read_csv('../input/covid19-russia-regions-cases/moscow_addresses.csv')



rus_bnd_gdf = gpd.read_file('../input/russia-geo-data/geo.json')
rus_df = rus_df.rename(columns={"Region/City": "Region", "Region/City-Eng": "Region_en", "Day-Confirmed": "DConf",

                   "Day-Deaths": "DDeath","Day-Recovered": "DRec","Confirmed": "Conf",

                   "Recovered": "Rec", "Deaths": "Death"})



rus_df.Date = pd.to_datetime(rus_df.Date, dayfirst=True)

print('Data date:', rus_df.iloc[-1]['Date'].strftime('%d, %b %Y'))
# Get current situation in regions

rus_df.Date = pd.to_datetime(rus_df.Date, dayfirst=True) 

rus_df['Act'] = rus_df.Conf - rus_df.Death - rus_df.Rec

rus_df['Death_rate'] = rus_df.Death / rus_df.Conf * 100

rus_df = rus_df[rus_df.Region != 'Diamond Princess']

cases = ['Conf', 'Death', 'Rec', 'Act', 'Death_rate']

rus_latest = rus_df.groupby('Region').apply(lambda df: df.loc[df.Date.idxmax()])

rus_latest = rus_latest.sort_values(by='Conf', ascending=False).reset_index(drop=True)

rus_latest = rus_latest[['Region', 'Region_en', 'Region_ID'] + cases]

# rus_latest
# Align rus_info_df Region to rus_df Region 

rename_dict = {

    'Республика Северная Осетия': 'Республика Северная Осетия - Алания',

    'Ямало-Ненецкий автономный округ': 'Ямало-Ненецкий АО',

    'Белгородская область': 'Белгород',

    'Республика Коми': 'Республика коми',

    'Камчатский край' : 'Камчатский край'

}



def rename(row):

    name = row['Region'][0]

    return pd.Series(rename_dict[name] if name in rename_dict else name)



# rus_info_df.Region = rus_info_df.reset_index().groupby('index').apply(rename).reset_index(drop=True)
# Prepare Rus geo data



# Combining with regions info

right = rus_info_df[['Region_ID', 'Population', 'Latitude', 'Longitude']]

rus_geo_df = rus_latest.merge(right, left_on='Region_ID', right_on='Region_ID', how='left')



# Confirmed per 100K

rus_geo_df['Conf_p100k'] = rus_geo_df.Conf / (rus_geo_df.Population / 100000)

rus_geo_df = rus_geo_df.drop('Population', axis=1)



# rus_geo_df.loc[np.isnan(rus_geo_df.Conf) | np.isnan(rus_geo_df.Latitude)]



# Manual fixes

rus_geo_df.loc[rus_geo_df.Region == 'Москва', 'Latitude'] = 55.7522

rus_geo_df.loc[rus_geo_df.Region == 'Москва', 'Longitude'] = 37.6220

rus_geo_df.loc[rus_geo_df.Region == 'Санкт-Петербург', 'Latitude'] = 59.8917

rus_geo_df.loc[rus_geo_df.Region == 'Санкт-Петербург', 'Longitude'] = 30.2673



rus_geo_df[['Region_en'] + cases + ['Conf_p100k']].style.background_gradient(cmap='Reds')
def embed_map(m, file_name):

    from IPython.display import IFrame

    m.save(file_name)

    return IFrame(file_name, width='100%', height='500px')
m_1 = folium.Map(

    location=[64.0914, 101.6016],

#     tiles='Stamen Toner',

    zoom_start=3

)



max_psize = 150000

min_psize = 20000

min_val = rus_geo_df.Conf.min()

max_val = rus_geo_df.Conf.max()



for i in range(len(rus_geo_df)):

    radius = min_psize + (rus_geo_df.Conf[i] - min_val) / (max_val - min_val) * (max_psize - min_psize)

    folium.Circle(

        radius=radius,

        location=[rus_geo_df.Latitude[i], rus_geo_df.Longitude[i]],

        popup=rus_geo_df.Region_en[i] + ' {}'.format(int(rus_geo_df.Conf[i])),

        color='crimson',

        fill=True,

    ).add_to(m_1)



# folium.LatLngPopup().add_to(m_1)



embed_map(m_1, 'm_1.html')
# Regions mapping

rus_gdf = gpd.GeoDataFrame(rus_geo_df, geometry=gpd.points_from_xy(rus_geo_df.Longitude, rus_geo_df.Latitude))

rus_gdf.crs = {'init': 'epsg:4326'}



rus_shape = rus_bnd_gdf[['NAME_1', 'TYPE_1', 'ID_1', 'geometry']]

rus_gdf = gpd.sjoin(rus_gdf, rus_shape, how="left", op='within')



# rus_gdf.loc[np.isnan(rus_gdf.ID_1)]

# rus_gdf.head()
m_3 = folium.Map(

    location=[64.0914, 101.6016],

    tiles='Stamen Toner',

    zoom_start=3

)



scale_min, scale_max = np.log(rus_gdf.Act.min() + 1), np.log(rus_gdf.Act.max() + 1)

colormap = linear.YlOrRd_09.scale(scale_min, scale_max)



def color_mapper(id):

    row = rus_gdf[rus_gdf.ID_1 == id].reset_index()

    if len(row) == 0:

        return scale_min

    return np.log(row.Act.iloc[0] + 1)



folium.GeoJson(

    rus_bnd_gdf,

    name='rusjson',

    style_function=lambda feature: {

        'fillColor': colormap(color_mapper(feature['properties']['ID_1'])),

        'color': 'black',

        'weight': 1,

        'dashArray': '5, 5',

        'fillOpacity': 0.9,

    }

).add_to(m_3)



for i in range(len(rus_gdf)):

    folium.Circle(

        radius=20000,

        location=[rus_gdf.Latitude[i], rus_gdf.Longitude[i]],

        popup=rus_gdf.Region_en[i] + ' Active: {}'.format(int(rus_gdf.Act[i])),

        color='crimson',

        fill=True,

    ).add_to(m_3)



# folium.LatLngPopup().add_to(m_3)



# colormap.caption = 'Active cases color scale'

# colormap.add_to(m_3)



embed_map(m_3, 'm_3.html')
m_4 = folium.Map(

    location=[64.0914, 101.6016],

    tiles='Stamen Toner',

    zoom_start=3

)



scale_min, scale_max = np.log(rus_gdf.Conf_p100k.min() + 1), np.log(rus_gdf.Conf_p100k.max() + 1)

colormap = linear.YlOrRd_09.scale(scale_min, scale_max)



def color_mapper(id):

    row = rus_gdf[rus_gdf.ID_1 == id].reset_index()

    if len(row) == 0:

        return scale_min

    return np.log(row.Conf_p100k.iloc[0] + 1)



folium.GeoJson(

    rus_bnd_gdf,

    name='rusjson',

    style_function=lambda feature: {

        'fillColor': colormap(color_mapper(feature['properties']['ID_1'])),

        'color': 'black',

        'weight': 1,

        'dashArray': '5, 5',

        'fillOpacity': 0.9,

    }

).add_to(m_4)



for i in range(len(rus_gdf)):

    folium.Circle(

        radius=20000,

        location=[rus_gdf.Latitude[i], rus_gdf.Longitude[i]],

        popup=rus_gdf.Region_en[i] + ' Conf. per 100k: {:6.2f}, Conf: {}'.format(rus_gdf.Conf_p100k[i], int(rus_gdf.Conf[i])),

        color='crimson',

        fill=True,

    ).add_to(m_4)





embed_map(m_4, 'm_4.html')
m_5 = folium.Map(

    location=[64.0914, 101.6016],

    tiles='Stamen Toner',

    zoom_start=3

)



scale_min, scale_max = np.log(rus_gdf.Death_rate.min() + 1), np.log(rus_gdf.Death_rate.max() + 1)

colormap = linear.YlOrRd_09.scale(scale_min, scale_max)



def color_mapper(id):

    row = rus_gdf[rus_gdf.ID_1 == id].reset_index()

    if len(row) == 0:

        return scale_min

    return np.log(row.Death_rate.iloc[0] + 1)



folium.GeoJson(

    rus_bnd_gdf,

    name='rusjson',

    style_function=lambda feature: {

        'fillColor': colormap(color_mapper(feature['properties']['ID_1'])),

        'color': 'black',

        'weight': 1,

        'dashArray': '5, 5',

        'fillOpacity': 0.9,

    }

).add_to(m_5)



for i in range(len(rus_gdf)):

    folium.Circle(

        radius=20000,

        location=[rus_gdf.Latitude[i], rus_gdf.Longitude[i]],

        popup=rus_gdf.Region_en[i] + ' Death Rate: {:6.2f}, Conf: {}'.format(rus_gdf.Death_rate[i], int(rus_gdf.Conf[i])),

        color='crimson',

        fill=True,

    ).add_to(m_5)





embed_map(m_5, 'm_5.html')
m_2 = folium.Map(

    location=[55.7522, 37.6220],

    tiles='Stamen Toner',

    zoom_start=10

)



for i in range(len(mos_addr_df)):

    folium.Circle(

        radius=50,

        location=[mos_addr_df.Latitude[i], mos_addr_df.Longitude[i]],

        popup=mos_addr_df.Address[i],

        color='crimson',

        fill=True,

    ).add_to(m_2)



HeatMap(mos_addr_df[['Latitude', 'Longitude']], radius=15).add_to(m_2)

    

embed_map(m_2, 'm_2.html')