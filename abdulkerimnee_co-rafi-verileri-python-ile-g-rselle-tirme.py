%matplotlib inline
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
pd.options.display.max_rows = 10
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
ulkeler=gpd.read_file('../input/ne-110m-admin-0-countries/ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp')
sehirler=gpd.read_file('../input/natural-earth/110m_cultural/ne_110m_populated_places_simple.shp')
nehir=gpd.read_file('../input/natural-earth/50m_physical/ne_50m_rivers_lake_centerlines.shp')
ulkeler.plot();# en basit haliyle
ulkeler.plot(figsize=(10,10));# çıktının boyutunu ayarlama
ax =ulkeler.plot(figsize=(10, 15))
ax.set_axis_off()#Dış çerveyi ve koordinatları(x,y) kaldırma
ulkeler.rename(columns={'pop_est':'Nüfus',
                      'continent':'Kıta',
                      'name':'Ülkeler','gdp_md_est':'Alan'},inplace=True)
ulkeler.head()#sütun isimlerini değiştirme
ulkeler=ulkeler[(ulkeler['Ülkeler'] != "Antarctica")]#Antarktika'yı kaldırma
ulkeler['GSYİH'] = ulkeler['Alan'] / ulkeler['Nüfus'] * 100#GSYİH sütununu oluşturma
ax = ulkeler.plot(figsize=(15, 15), column='GSYİH')
ax.set_axis_off()#GSYİH sütununa göre tematik harita yapma
ax = ulkeler.plot(figsize=(15, 15),scheme='quantiles',cmap='OrRd', column='GSYİH')
ax.set_axis_off()#GSYİH sütununa göre tematik harita yapma
import geoplot
import geoplot.crs as gcrs
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={
    'projection': gcrs.Orthographic(central_latitude=41.935, central_longitude=29.773)
})
geoplot.choropleth(ulkeler, hue='GSYİH', projection=gcrs.Orthographic(), ax=ax,
                   cmap='magma', linewidth=0.5, edgecolor='white')
ax.set_global()
ax.outline_patch.set_visible(True)
ax.coastlines();
from cartopy import crs as ccrs
ulkeler.crs = {'init' :'epsg:4326'}
# referans sistemi tanımlama.
crs = ccrs.AlbersEqualArea()
crs_proj4 = crs.proj4_init
ulkeler_alan_koruyan = ulkeler.to_crs(crs_proj4)
ulkeler_alan_koruyan.crs#referans sistemi sorgulama
ulkeler_alan_koruyan.plot();#geopandas ile görselleştirme
fig, ax = plt.subplots(subplot_kw={'projection': crs})
ax.add_geometries(ulkeler_alan_koruyan['geometry'], crs=crs);#cartopy ile görselleştirme
fig, ax = plt.subplots(subplot_kw={'projection': crs})
ulkeler_alan_koruyan['geometry'].plot(ax=ax);#hem cartopy hemde geopandas ile görselleştirme
import folium
m = folium.Map(location=[41.935,29.773],zoom_start=4)
m
m = folium.Map(location=[41.935,29.773],zoom_start=4)
folium.TileLayer('Stamen Terrain').add_to(m);
m
m = folium.Map(
    location=[41.935,29.773],
    zoom_start=4,
    tiles='Stamen Terrain'
)



folium.Marker([41.0102073, 28.9609019], popup='Istanbul', tooltip='Istanbul').add_to(m)
folium.Marker([39.9232304,32.9398666], popup='Ankara', tooltip='Ankara').add_to(m)

m
m = folium.Map(
    location=[41.935,29.773],
    zoom_start=4,
    tiles='Stamen Terrain'
)

m.add_child(folium.ClickForMarker())
m.add_child(folium.LatLngPopup())

m
m = folium.Map(
    location=[41.935,29.773],
    zoom_start=5,
    tiles='Stamen Toner'
)
folium.Circle(
    radius=500,
    location=[41.0102073, 28.9609019],
    popup='Istanbul',
    color='crimson',
    fill=False,
).add_to(m)

folium.CircleMarker(
    location=[39.9232304,32.9398666],
    radius=50,
    popup='Ankara',
    color='#3186cc',
    fill=True,
    fill_color='#3186cc'
).add_to(m)


m
m = folium.Map([0, 0], zoom_start=1)
folium.Choropleth(geo_data=ulkeler, data=ulkeler, columns=['iso_a3', 'GSYİH'],
             key_on='feature.properties.iso_a3', fill_color='OrRd', highlight=True).add_to(m)
m#tematik harita