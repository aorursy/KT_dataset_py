%matplotlib inline
import pandas as pd
import geopandas as gpd
ulkeler=gpd.read_file('../input/ne-110m-admin-0-countries/ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp')
ulkeler.head()
ulkeler.plot();
ulkeler.geometry
type(ulkeler.geometry)
ulkeler.geometry.area
ulkeler['pop_est'].mean()
turkiye=ulkeler[ulkeler['name'] == 'Turkey']
turkiye.plot();
sehirler=gpd.read_file('../input/natural-earth/110m_cultural/ne_110m_populated_places_simple.shp')
type(sehirler.geometry[0])
sehirler.plot();
nehir=gpd.read_file('../input/natural-earth/50m_physical/ne_50m_rivers_lake_centerlines.shp')
type(nehir.geometry[0])
nehir.plot();
type(ulkeler.geometry[0])
ulkeler.plot();
type(ulkeler.geometry[0])
ax = ulkeler.plot(edgecolor='k', facecolor='none', figsize=(15, 10))
nehir.plot(ax=ax)
sehirler.plot(ax=ax, color='red')
ax.set(xlim=(0, 50), ylim=(25, 75));
from shapely.geometry import Point, Polygon, LineString
p = Point(0, 0)
p
polygon = Polygon([(1, 1), (2,2), (2, 1)])
polygon
polygon.area#alan sorgulama
df = pd.DataFrame(
    {'Sehirler': ['İstanbul', 'Ankara', 'Konya','Erzurum'],
     'Boylam':[29.00806,32.87110,32.49760,41.29210],
     'Enlem':[41.10694,39.90130,37.8570,39.88450]})

df
gdf = gpd.GeoDataFrame(
    df, geometry=gpd.points_from_xy(df.Boylam, df.Enlem))
gdf.head()
ax = ulkeler.plot(edgecolor='k', facecolor='none', figsize=(15, 10))
nehir.plot(ax=ax)
gdf.plot(ax=ax, color='red')
ax.set(xlim=(0, 50), ylim=(25, 75));