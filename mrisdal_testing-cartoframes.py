!pip install cartoframes
from pandas import read_csv

from geopandas import GeoDataFrame, points_from_xy



remote_file_path = 'http://data.sfgov.org/resource/wg3w-h783.csv'



df = read_csv(remote_file_path)



# Clean latitude and longitude values that are NaN

df = df[df.longitude == df.longitude]

df = df[df.latitude == df.latitude]



gdf = GeoDataFrame(df, geometry=points_from_xy(df['longitude'], df['latitude']))

gdf.head()
from cartoframes.viz import Layer



Layer(gdf)