%matplotlib inline

import pandas as pd

import geopandas as gpd

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
ulkeler= gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
ulkeler.head()
ulkeler.crs
ulkeler=ulkeler[(ulkeler['name'] != "Antarctica")]
ulkeler_mercator = ulkeler.to_crs(epsg=3395)
ulkeler_mercator.crs
ax =ulkeler_mercator.plot(figsize=(10,10))

ax.set_title("Mercator");
ax =ulkeler.plot(figsize=(10,10))

ax.set_title("WGS84");
df = pd.DataFrame(

    {'Sehirler': ['İstanbul', 'Ankara', 'Konya','Erzurum'],

     'Boylam':[29.00806,32.87110,32.49760,41.29210],

     'Enlem':[41.10694,39.90130,37.8570,39.88450]})

gdf = gpd.GeoDataFrame(

    df, geometry=gpd.points_from_xy(df.Boylam, df.Enlem))
gdf.head()
sorgu=gdf.crs

print(sorgu)
gdf.crs = {'init' :'epsg:4326'}
gdf.crs
ulkeler_mercator=ulkeler_mercator.to_crs(ulkeler.crs)
ulkeler_mercator.crs