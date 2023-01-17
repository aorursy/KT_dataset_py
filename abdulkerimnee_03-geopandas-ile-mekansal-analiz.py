%matplotlib inline
import pandas as pd
import geopandas as gpd
ulkeler=gpd.read_file('../input/ne-110m-admin-0-countries/ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp')
sehirler=gpd.read_file('../input/natural-earth/110m_cultural/ne_110m_populated_places_simple.shp')
nehir=gpd.read_file('../input/natural-earth/50m_physical/ne_50m_rivers_lake_centerlines.shp')
turkiye=ulkeler.loc[ulkeler['name'] == 'Turkey', 'geometry'].squeeze()
turkiye
istanbul=sehirler.loc[sehirler['name'] == 'Istanbul', 'geometry'].squeeze()
ankara=sehirler.loc[sehirler['name'] == 'Ankara', 'geometry'].squeeze()
baku=sehirler.loc[sehirler['name']=='Baku','geometry'].squeeze()
from shapely.geometry import LineString
çizgi= LineString([istanbul,ankara,baku])
çizgi
gpd.GeoSeries([turkiye, istanbul, ankara,baku,çizgi]).plot(cmap='RdYlGn',figsize=(15,10));
ankara.within(turkiye)
turkiye.contains(ankara)
baku.within(turkiye)
turkiye.contains(çizgi)
çizgi.intersects(turkiye)
ulkeler.contains(ankara)
ulkeler[ulkeler.contains(ankara)]
amazon =nehir[nehir['name'] == 'Amazonas'].geometry.squeeze()
amazon
ulkeler[ulkeler.crosses(amazon)]
gpd.GeoSeries([turkiye,ankara.buffer(1)]).plot(alpha=0.8, cmap='tab10');
ankara.buffer(1).intersection(turkiye)#kesişimi
ankara.buffer(1).union(turkiye)#birleşimi
fark=ankara.buffer(1).difference(turkiye)#farkı
print(fark)# alan boş sonucu vermiştir
afrika_kıtası=ulkeler[ulkeler['continent'] == 'Africa']
afrika_kıtası.plot();
afrika=afrika_kıtası.unary_union
afrika
ulkeler.head()
sehirler.head()
sehirler2= sehirler[sehirler['name'].isin(['Bern', 'Brussels', 'London', 'Paris'])].copy()
sehirler2['iso_a3'] = ['CHE', 'BEL', 'GBR', 'FRA']
sehirler2.head()
ulkeler2=ulkeler[['iso_a3', 'name', 'continent']]
ulkeler2.head()
birleşme=sehirler2.merge(ulkeler2, on='iso_a3')
birleşme.head()