!pip install geopandas -q
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import geopandas as gpd
import folium
import re
# получаем населенные пункты
file = '/kaggle/input/tuva-populaton/populat_tuva.xlsx'
geodata = pd.read_excel(file, sheet_name=0, header=0, index_col='Unnamed: 0')
def name_ch(name):
    return re.sub(r'район', 'кожуун', name)
geodata['area'] = geodata['area'].apply(lambda x : name_ch(x))
# получаем json map
geojson = '/kaggle/input/tuva-geojson/koguun.geojson'
geo_map = gpd.read_file(geojson)
# Контурная карта
geo_map = geo_map.to_crs({'init' :'epsg:3857'})       
geo_map.plot(linewidth=0.5, edgecolor='black', color = 'white', figsize=[15,15]).set_title('Кожууны', fontsize=18)
# Заливка
geo = geo_map.to_crs({'init' :'epsg:3857'})
geo.plot(column = 'name', linewidth=0.5, cmap='plasma', legend=True, figsize=[15,15])
# folium
points = (geodata.lat.fillna(0),geodata.lot.fillna(0))
lat = points[0]
long = points[1]

# Map
map_tuva = folium.Map(location=[51.719082, 94.433983],width=800, height=600,max_zoom=7, tiles='Stamen Toner')
pop_title = geodata['population_2010']


style = {'fillColor': '#f7dfc8', 'lineColor': '#000'}

gj = folium.GeoJson(geo, control=True, smooth_factor=0, style_function=lambda x: style)
gj.add_to(map_tuva)

for la,lo in zip(lat,long):
    folium.CircleMarker(
        location=[la,lo],
        radius=1,
        #popup='Population',
        color='red',
        #fill=True,
        #fill_color='red'
    ).add_to(map_tuva)
    
map_tuva
