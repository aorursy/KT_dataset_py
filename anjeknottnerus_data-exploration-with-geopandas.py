#importing packages 

import pandas as pd

import geopandas as gpd

import matplotlib.pyplot as plt

from shapely.geometry import Point, Polygon # Shapely for converting latitude/longtitude to geometry

import seaborn as sns

sns.set(rc={'figure.figsize':(20,10)})
def load_data(file):

    df = pd.read_csv(file)

    print("The data is loaded...")

    return df

    

def prepare_data(df):

    df = df.drop(columns=['CatalogNumber', 'DataProvider', 'ScientificName'

                          , 'DepthMethod', 'Locality','LocationAccuracy'

                          , 'SurveyID', 'Repository', 'IdentificationQualifier','EventID'

                          , 'SamplingEquipment', 'RecordType', 'SampleID'])

    print("The following columns are dropped: ['CatalogNumber', 'DataProvider', 'ScientificName', 'DepthMethod', 'Locality','LocationAccuracy','SurveyID','Repository', 'IdentificationQualifier', 'EventID', 'SamplingEquipment', 'RecordType', 'SampleID'] ")

    return df

        

def clean_data(df):

    print("The dataset contains", len(df), "rows")

    print("The unavailable rows are dropped...")

    df = df.iloc[1:]

    df = df.dropna()

    print("The dataset contains", len(df), "rows")

    return df   

 

#dataframe to geodataframe 

def df_to_gdf(df):

    coordinates = df[['latitude', 'longitude']].astype(float).values

    coordinates = pd.DataFrame(data=coordinates)

    coordinates.columns = ['latitude', 'longitude']

    coordinates = [Point(xy) for xy in zip(coordinates.longitude, coordinates.latitude)]

    geo_df = gpd.GeoDataFrame(df, crs={'init': 'epsg:4326'}, geometry=coordinates)

    return geo_df
data = load_data('../input/deep-sea-corals/deep_sea_corals.csv')

data = prepare_data(data)
#loading worldmap

world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

base = world.plot(color='white', edgecolor='black')



#absolute value of depth in meters

data["DepthInMeters"] = data["DepthInMeters"].abs() 



#subsetting the corals above 200 meters and below 1000 meters

ss_min_depth = data[data['DepthInMeters'] < 200]

ss_max_depth = data[data['DepthInMeters'] > 1000]



#to geo dataframe

gdf_min_depth = df_to_gdf(ss_min_depth)

gdf_max_depth = df_to_gdf(ss_max_depth)



gdf_min_depth.plot(ax=base, marker='o', color='red', markersize=0.5)

gdf_max_depth.plot(ax=base, marker='o', color='green', markersize=0.5)



plt.show()
#loading worldmap

world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

base = world.plot(color='white', edgecolor='black')



data["ObservationYear"] =  pd.DatetimeIndex(data['ObservationDate']).year



#subsetting different observation years

ss_2012 = data[data["ObservationYear"] == 2012]

ss_2013 = data[data["ObservationYear"] == 2013]

ss_2014 = data[data["ObservationYear"] == 2014]

ss_2015 = data[data["ObservationYear"] == 2015]



#to geo dataframe

gdf_2012 = df_to_gdf(ss_2012)

gdf_2013 = df_to_gdf(ss_2013)

gdf_2014 = df_to_gdf(ss_2014)

gdf_2015 = df_to_gdf(ss_2015)



gdf_2012.plot(ax=base, marker='o', color='green', markersize=0.5)

gdf_2013.plot(ax=base, marker='o', color='red', markersize=0.5)

gdf_2014.plot(ax=base, marker='o', color='yellow', markersize=0.5)

gdf_2015.plot(ax=base, marker='o', color='blue', markersize=0.5)



plt.show()