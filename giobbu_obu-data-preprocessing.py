import os 

# Disable warnings, set Matplotlib inline plotting and load Pandas package

import warnings

warnings.filterwarnings('ignore')

import pandas as pd

import numpy as np

from datetime import datetime

from pytz import timezone

from dateutil import tz

from datetime import datetime, timedelta

import geojson

import geopandas as gpd  

from fiona.crs import from_epsg

import os, json

from shapely.geometry import shape, Point, Polygon, MultiPoint



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

%matplotlib inline

import matplotlib.pyplot as plt

import osmnx as ox





import os 

# Disable warnings, set Matplotlib inline plotting and load Pandas package

import warnings

warnings.filterwarnings('ignore')

import pandas as pd

import numpy as np

#pd.options.display.mpl_style = 'default'

from datetime import datetime

import numpy as np

from datetime import datetime, timedelta

from pytz import timezone

from dateutil import tz

import geojson

import geopandas as gpd

from fiona.crs import from_epsg

import os, json

from shapely.geometry import shape, Point, Polygon, MultiPoint

%matplotlib inline

import matplotlib.pyplot as plt

from geopandas.tools import sjoin

from sklearn.neighbors import KernelDensity

from sklearn.model_selection import train_test_split

import matplotlib.cm as cm



import folium
def getDuplicateColumns(df):

    '''

    Get a list of duplicate columns.

    It will iterate over all the columns in dataframe and find the columns whose contents are duplicate.

    :param df: Dataframe object

    :return: List of columns whose contents are duplicates.

    '''

    duplicateColumnNames = set()

    # Iterate over all the columns in dataframe

    for x in range(df.shape[1]):

        # Select column at xth index.

        col = df.iloc[:, x]

        # Iterate over all the columns in DataFrame from (x+1)th index till end

        for y in range(x + 1, df.shape[1]):

            # Select column at yth index.

            otherCol = df.iloc[:, y]

            # Check if two columns at x 7 y index are equal

            if col.equals(otherCol):

                duplicateColumnNames.add(df.columns.values[y])

 

    return list(duplicateColumnNames)
df_bruxelles = gpd.read_file('../input/belgium-obu/Bruxelles_streets.json')

print('Bruxelles total number of streets '+str(df_bruxelles.shape[0]))





polygons = df_bruxelles

m = folium.Map([50.85045, 4.34878], zoom_start=12, tiles='cartodbpositron')

folium.GeoJson(polygons).add_to(m)

m
DF_15 = pd.read_csv('../input/belgium-obu/Bxl_15.csv', header=None)

DF_15.columns = ['datetime','street_id','count','vel']

nRow_15, nCol_15 = DF_15.shape



DF_30 = pd.read_csv('../input/belgium-obu/Bxl_30.csv', header=None)

DF_30.columns = ['datetime','street_id','count','vel']

nRow_30, nCol_30 = DF_30.shape



DF_60 = pd.read_csv('../input/belgium-obu/Bxl_60.csv', header=None)

DF_60.columns = ['datetime','street_id','count','vel']

nRow_60, nCol_60 = DF_60.shape





print(f'in BXL 15 min there are {nRow_15} rows and {nCol_15} columns')

print(f'in BXL 30 min there are {nRow_30} rows and {nCol_30} columns')

print(f'in BXL 60 min there are {nRow_60} rows and {nCol_60} columns')
os.stat('../input/belgium-obu/Bxl_60.csv').st_size/(1024*1024)
table_15 = DF_15.pivot_table(index='datetime', columns='street_id')['count'].fillna(0)

table_vel_15 = DF_15.pivot_table(index='datetime', columns='street_id')['vel'].fillna(0)

print(table_15.shape)

print('')



table_30 = DF_30.pivot_table(index='datetime', columns='street_id')['count'].fillna(0)

table_vel_30 = DF_30.pivot_table(index='datetime', columns='street_id')['vel'].fillna(0)

print(table_30.shape)

print('')



table_60 = DF_60.pivot_table(index='datetime', columns='street_id')['count'].fillna(0)

table_vel_60 = DF_60.pivot_table(index='datetime', columns='street_id')['vel'].fillna(0)

print(table_60.shape)

print('')
list_duplicates = getDuplicateColumns(table_vel_60)

print(len(list_duplicates))
file_name = 'Flow_BXL_street_15min'

print(file_name)

table_15 = table_15.reset_index().drop(list_duplicates, axis=1)

table_15.to_csv(file_name + '.csv',index=False)

print(table_15.shape)

print('')



file_name = 'Velocity_BXL_street_15min'

print(file_name)

table_vel_15 = table_vel_15.reset_index().drop(list_duplicates, axis=1)

table_vel_15.to_csv(file_name + '.csv',index=False)

print(table_vel_15.shape)

print('')

print('')





file_name = 'Flow_BXL_street_30min'

print(file_name)

table_30 = table_30.reset_index().drop(list_duplicates, axis=1)

table_30.to_csv(file_name + '.csv',index=False)

print(table_30.shape)

print('')



file_name = 'Velocity_BXL_street_30min'

print(file_name)

table_vel_30 = table_vel_30.reset_index().drop(list_duplicates, axis=1)

table_vel_30.to_csv(file_name + '.csv',index=False)

print(table_vel_30.shape)

print('')

print('')





file_name = 'Flow_BXL_street_60min'

print(file_name)

table_60 = table_60.reset_index().drop(list_duplicates, axis=1)

table_60.to_csv(file_name + '.csv',index=False)

print(table_60.shape)

print('')



file_name = 'Velocity_BXL_street_60min'

print(file_name)

table_vel_60 = table_vel_60.reset_index().drop(list_duplicates, axis=1)

table_vel_60.to_csv(file_name + '.csv',index=False)

print(table_vel_60.shape)
df_belgium = gpd.read_file('../input/belgium-obu/Belgium_streets.json')

print('Belgium total number of highways '+str(df_belgium.shape[0]))



m = folium.Map([50.85045, 4.34878], zoom_start=9, tiles='cartodbpositron')

folium.GeoJson(df_belgium).add_to(m)

m
DF_15 = pd.read_csv('../input/belgium-obu/Bel_15.csv', header=None)

DF_15.columns = ['datetime','street_id','count','vel']

nRow_15, nCol_15 = DF_15.shape



DF_30 = pd.read_csv('../input/belgium-obu/Bel_30.csv', header=None)

DF_30.columns = ['datetime','street_id','count','vel']

nRow_30, nCol_30 = DF_30.shape



DF_60 = pd.read_csv('../input/belgium-obu/Bel_60.csv', header=None)

DF_60.columns = ['datetime','street_id','count','vel']

nRow_60, nCol_60 = DF_60.shape



print(f'in BEL 15 min there are {nRow_15} rows and {nCol_15} columns')

print(f'in BEL 30 min there are {nRow_30} rows and {nCol_30} columns')

print(f'in BEL 60 min there are {nRow_60} rows and {nCol_60} columns')
table_15 = DF_15.pivot_table(index='datetime', columns='street_id')['count'].fillna(0)

table_vel_15 = DF_15.pivot_table(index='datetime', columns='street_id')['vel'].fillna(0)



table_30 = DF_30.pivot_table(index='datetime', columns='street_id')['count'].fillna(0)

table_vel_30 = DF_30.pivot_table(index='datetime', columns='street_id')['vel'].fillna(0)



table_60 = DF_60.pivot_table(index='datetime', columns='street_id')['count'].fillna(0)

table_vel_60 = DF_60.pivot_table(index='datetime', columns='street_id')['vel'].fillna(0)

print(table_60.shape)

print('')
DATAFRAME = table_vel_60

list_duplicates = getDuplicateColumns(DATAFRAME)

print(len(list_duplicates))
file_name = 'Flow_BEL_street_15min'

print(file_name)

table_15 = table_15.reset_index().drop(list_duplicates, axis=1)

table_15.to_csv(file_name + '.csv',index=False)

print(table_15.shape)

print('')



file_name = 'Velocity_BEL_street_15min'

print(file_name)

table_vel_15 = table_vel_15.reset_index().drop(list_duplicates, axis=1)

table_vel_15.to_csv(file_name + '.csv',index=False)

print(table_vel_15.shape)

print('')

print('')





file_name = 'Flow_BEL_street_30min'

print(file_name)

table_30 = table_30.reset_index().drop(list_duplicates, axis=1)

table_30.to_csv(file_name + '.csv',index=False)

print(table_30.shape)

print('')



file_name = 'Velocity_BEL_street_30min'

print(file_name)

table_vel_30 = table_vel_30.reset_index().drop(list_duplicates, axis=1)

table_vel_30.to_csv(file_name + '.csv',index=False)

print(table_vel_30.shape)

print('')

print('')





file_name = 'Flow_BEL_street_60min'

print(file_name)

table_60 = table_60.reset_index().drop(list_duplicates, axis=1)

table_60.to_csv(file_name + '.csv',index=False)

print(table_60.shape)

print('')



file_name = 'Velocity_BEL_street_60min'

print(file_name)

table_vel_60 = table_vel_60.reset_index().drop(list_duplicates, axis=1)

table_vel_60.to_csv(file_name + '.csv',index=False)

print(table_vel_60.shape)