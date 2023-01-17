import os, sys
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from geopandas.tools import sjoin
from matplotlib import pyplot as plt
import warnings
warnings.simplefilter(action='ignore')
print(os.listdir("../input"))
# Lets look at the actual data now

data_folder = "../input/data-science-for-good/cpe-data/"

#data_folder = "C:/Users/Lilianne/Downloads/cpe-data/"
dirs = os.listdir( data_folder )

deparment_list = []

for filename in os.listdir(data_folder):
    if os.path.isdir(os.path.join(data_folder,filename)):
        deparment_list.append(filename)

print(deparment_list)    
department = "Dept_11-00091"

dirs = os.listdir( data_folder + department + "/")

for file in dirs:
    print(file)
department = "Dept_23-00089"

dirs = os.listdir( data_folder + department + "/")

for file in dirs:
    print(file)
department = "Dept_35-00103"

dirs = os.listdir( data_folder + department + "/")

for file in dirs:
    print(file)
department = "Dept_37-00027"

dirs = os.listdir( data_folder + department + "/")

for file in dirs:
    print(file)
department = "Dept_37-00049"

dirs = os.listdir( data_folder + department + "/")

for file in dirs:
    print(file)
department = "Dept_49-00009"

dirs = os.listdir( data_folder + department + "/")

for file in dirs:
    print(file)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
department = "37-00049"

dirs = os.listdir( data_folder + "Dept_" + department + "/" + department + "_ACS_data/")

for file in dirs:
    print(file)
census_race_df = pd.read_csv(data_folder + "Dept_" + department + "/" + department + "_ACS_data/" + department + "_ACS_race-sex-age/ACS_15_5YR_DP05_with_ann.csv").iloc[0:].reset_index(drop=True)

police_incident_df = pd.read_csv(data_folder + "Dept_" + department + "/" + department + "_UOF-P_2016_prepped.csv").iloc[1:].reset_index(drop=True)
census_race_df.head()
police_incident_df.head()
department = "37-00027"

dirs = os.listdir( data_folder + "Dept_" + department + "/" + department + "_ACS_data/")

census_race_df_3700027 = pd.read_csv(data_folder + "Dept_" + department + "/" + department + "_ACS_data/" + department + "_ACS_race-sex-age/ACS_15_5YR_DP05_with_ann.csv").iloc[0:].reset_index(drop=True)

police_incident_df_3700027 = pd.read_csv(data_folder + "Dept_" + department + "/" + department + "_UOF-P_2014-2016_prepped.csv").iloc[1:].reset_index(drop=True)
census_race_df_3700027.head()
police_incident_df_3700027.head()
department = "35-00103"

dirs = os.listdir( data_folder + "Dept_" + department + "/" + department + "_ACS_data/")

census_race_df_3500103 = pd.read_csv(data_folder + "Dept_" + department + "/" + department + "_ACS_data/" + department + "_ACS_race-sex-age/ACS_15_5YR_DP05_with_ann.csv").iloc[0:].reset_index(drop=True)

police_incident_df_3500103 = pd.read_csv(data_folder + "Dept_" + department + "/" + department + "_UOF-OIS-P_prepped.csv").iloc[1:].reset_index(drop=True)
census_race_df_3500103.head()
police_incident_df_3500103.head()
census_shp_gdf_3500103 = gpd.read_file("../input/censusshapes3500103/tl_2010_37119_tract10.shp")
census_shp_gdf_3500103.head()
latlon_exists_index = police_incident_df_3500103[['LOCATION_LATITUDE','LOCATION_LONGITUDE']].dropna().index

police_incident_df_3500103 = police_incident_df_3500103.iloc[latlon_exists_index].reset_index(drop=True)

police_incident_df_3500103['LOCATION_LATITUDE'] = (police_incident_df_3500103['LOCATION_LATITUDE']
                                         .astype('float'))
police_incident_df_3500103['LOCATION_LONGITUDE'] = (police_incident_df_3500103['LOCATION_LONGITUDE']
                                         .astype('float'))

# important to check if order in Shapefile is Point(Longitude,Latitude)
police_incident_df_3500103['geometry'] = (police_incident_df_3500103
                                .apply(lambda x: Point(x['LOCATION_LONGITUDE'],
                                                       x['LOCATION_LATITUDE']), 
                                       axis=1))
police_incident_gdf_3500103 = gpd.GeoDataFrame(police_incident_df_3500103, geometry='geometry')
police_incident_gdf_3500103.crs = {'init' :'epsg:4326'}
police_incident_gdf_3500103.head()
# number of rows and number of columns in our dataset

police_incident_gdf_3500103.shape
max_1=police_incident_df_3500103.max()
min_1=police_incident_df_3500103.min()

print(min_1['INCIDENT_DATE'], max_1['INCIDENT_DATE'])

## plot
f, ax = plt.subplots(1, figsize=(10, 12))
census_shp_gdf_3500103.plot(ax=ax)
police_incident_gdf_3500103.plot(ax=ax, marker='*', color='black', markersize=15)
plt.title("Incident Locations and Census Tracts")
plt.show()