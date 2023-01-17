import numpy as np

import pandas as pd

import geopandas as gpd

import matplotlib.pyplot as plt

import matplotlib.colors

from pylab import rcParams

import seaborn as sns

from pathlib import Path

import os

import folium

from folium.plugins import MarkerCluster

import shapely

from shapely.geometry import Point

%matplotlib inline
# understanding column names

pd.read_csv('../input/crime-data-in-brazil/RDO_column_description.csv')
# list of csv files that will be used as input data

input_dir = '../input/crime-data-in-brazil/'

csv_list = sorted(list(os.listdir(input_dir)))

del csv_list[-5::]

for file in csv_list:

    print(file)
# columns to drop

cols = ['NUM_BO', 'ANO_BO', 'ID_DELEGACIA', 'NOME_DEPARTAMENTO',

        'NOME_SECCIONAL', 'DELEGACIA', 'DESDOBRAMENTO', 'CIDADE',

        'NOME_DEPARTAMENTO_CIRC', 'NOME_SECCIONAL_CIRC','NOME_DELEGACIA_CIRC',

        'MES', 'ANO', 'CONT_PESSOA', 'FLAG_STATUS', 'Unnamed: 30',

        'Unnamed: 31', 'Unnamed: 32', 'Unnamed: 33', 'Unnamed: 34',

        'Unnamed: 35', 'Unnamed: 36', 'Unnamed: 37', 'Unnamed: 38',

        'Unnamed: 39', 'Unnamed: 40', 'Unnamed: 41', 'Unnamed: 42',

        'Unnamed: 43', 'Unnamed: 44', 'Unnamed: 45', 'Unnamed: 46',

        'Unnamed: 47', 'Unnamed: 48', 'Unnamed: 49', 'Unnamed: 50',

        'Unnamed: 51', 'Unnamed: 52']



# some lat/long values are redacted and replaced with this str

drop_term = ['Informação restrita']



# data loading functions

def csv_to_df(csv):

    """ takes csv file and returns df """

    df = pd.DataFrame(pd.read_csv(input_dir + csv))

    return df



def drop_cols(df):

    """ deletes columns in cols dict from input df """

    for col in cols:

        if col in df.columns:

           del df[col]

    return df



# data cleaning/processing functions

def drop_redacted(df):

    df = df[~df['LATITUDE'].str.contains('|'.join(drop_term), na=False)]

    return df



def coord_floats(df):

    """ convert str long, lat values to floats """    

    df['LONGITUDE'] = df.apply(lambda row: float(row['LONGITUDE']), axis=1)

    df['LATITUDE'] = df.apply(lambda row: float(row['LATITUDE']), axis=1)

    return df
# load csv files into dict as dataframes, drop unnecessary columns from dataframes

df_dict = []



for file in csv_list:

    df = csv_to_df(file)

    df = drop_cols(df)

    df_dict.append(df)
# assign names to dataframes

df_2007 = df_dict[0].append(df_dict[1], ignore_index=True)

df_2008 = df_dict[2].append(df_dict[3], ignore_index=True)

df_2009 = df_dict[4].append(df_dict[5], ignore_index=True)

df_2010 = df_dict[6].append(df_dict[7], ignore_index=True)

df_2011 = df_dict[8].append(df_dict[9], ignore_index=True)

df_2012 = df_dict[10].append(df_dict[11], ignore_index=True)

df_2013 = df_dict[12].append(df_dict[13], ignore_index=True)

df_2014 = df_dict[14].append(df_dict[15], ignore_index=True)

df_2015 = df_dict[16]

df_2016 = df_dict[17]



dataframes = [df_2007, df_2008, df_2009, df_2010, df_2011, df_2012, df_2013, df_2014, df_2015, df_2016]
# checking for NaN coordinate values

def percent_nan(list):

    year = 2007

    for df in list:

        nan_lat = "{:.0%}".format((sum(df['LATITUDE'].isnull())/len(df)))

        nan_long = "{:.0%}".format((sum(df['LONGITUDE'].isnull())/len(df)))

        print(year, ':', nan_lat, 'NaN latitude values, ', nan_long, 'NaN longitude values')

        year += 1

        

percent_nan(dataframes)
# cleaning coordinate data for 2011 dataframe

df_2011 = drop_redacted(df_2011)

df_2011 = df_2011.dropna(subset=['LATITUDE', 'LONGITUDE'])

df_2011 = coord_floats(df_2011)

df_2011 = df_2011.reset_index(drop=True)
# check for NaN coordinate values in 2011 dataframe

print('NaN values left =', sum(df_2011['LATITUDE'].isnull()))
# create base map

sp_map = folium.Map(location=[-23.5505, -46.6333], zoom_start=10, tiles='CartoDB dark_matter')
# import Sao Paulo municipality boundary polygons from geojson file

sp_geo = gpd.read_file('../input/geojson/saopaulo.json')



# select after first 5, rows 1-4 are not municipalities

sp_geo = sp_geo.iloc[5:]

sp_geo = sp_geo.reset_index(drop=True)



# check quantity of municipalities in sao paulo, should be 645

print(len(sp_geo))
# functions to assign municipality to each row in df based on coordinate values

def check_polygon(point):

    """ 

    checks if input point is in each of polygons in sp_geo geodataframe 

    returns municipality name corresponding to polygon containg point

    """

    for index, row in sp_geo.iterrows():

        polygon = sp_geo.loc[index]['geometry']

        if polygon.contains(point) == False:

            continue

        else:

            if polygon.contains(point) == True:

                muni = sp_geo.loc[index]['NOMEMUNICP']

                return muni



def get_muni(long_series, lat_series):

    """ 

    takes series of long/lat values and series of geojson polygons

    returns list of corresponding municipalities where point is in polygon

    """

    points = [Point(xy) for xy in zip(long_series.values, lat_series.values)]

    muni_list = [check_polygon(x) for x in points]

    return muni_list
# randomly sample n = 35000 random rows from 2011 dataframe

# df_2011 = df_2011.reset_index()

df_2011_sample = df_2011.sample(1)
# assign municipality for each row in sampled df_2011 sample to ['MUNI'] column

new_cols = df_2011_sample.columns.tolist() + ['MUNI']

df_2011_sample = df_2011_sample.reindex(columns = new_cols)



muni_list = get_muni(df_2011_sample['LONGITUDE'], df_2011_sample['LATITUDE'])

muni_series= pd.Series(muni_list)

df_2011_sample['MUNI'] = muni_series

df_2011_sample.head()
# create chloropleth map of crime coordinate density in Sao Paulo municipalities

# sp_map.chloropleth(geo_data=sp_geo,

#                    data= ,

#                    columns= ,

#                    key_on= ,

#                    fill_color= ,

#                    fill_opacity=0.7,

#                    line_opacity=0.2,

#                    legend_name='Crime Event Density')