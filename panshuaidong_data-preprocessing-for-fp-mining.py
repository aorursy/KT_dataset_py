import numpy as np
import geopandas as gpd
from geopandas.tools import sjoin
from matplotlib import pyplot as plt
import pandas as pd
from shapely.geometry import Point
import os
import seaborn as sns
import folium
from folium import plugins
import geopandas as gpd
import plotly.graph_objs as go
import plotly.plotly as py
from shapely.geometry import Point


# Any results you write to the current directory are saved as output.
os.listdir("../input/data-science-for-good/cpe-data")
force_df = pd.read_csv('../input/data-science-for-good/cpe-data//Dept_37-00027'+
                         '/37-00027_UOF-P_2014-2016_prepped.csv')
force_clean_df = force_df.loc[1:].reset_index(drop=True)
force_clean_df ['LOCATION_LONGITUDE']= pd.to_numeric(force_clean_df['LOCATION_LONGITUDE'], downcast='float') 
force_clean_df ['LOCATION_LATITUDE']= pd.to_numeric(force_clean_df['LOCATION_LATITUDE'], downcast='float') 
force_clean_df = force_clean_df[np.isfinite(force_clean_df['LOCATION_LONGITUDE'])]
force_clean_df=force_clean_df[force_clean_df['LOCATION_LONGITUDE']!=0].reset_index(drop=True)
foo=lambda x: Point(x['LOCATION_LONGITUDE'],x['LOCATION_LATITUDE'])
force_clean_df['geometry'] = (force_clean_df.apply(foo, axis=1))
force_clean_df = gpd.GeoDataFrame(force_clean_df, geometry='geometry')
force_clean_df.crs = {'init' :'epsg:4326'}
police_df_Austin = gpd.read_file('../input/data-science-for-good/cpe-data/'
                                 +'Dept_37-00027/37-00027_Shapefiles/APD_DIST.shp')
police_df_Austin.crs = {'init' :'esri:102739'}
police_df_Austin = police_df_Austin.to_crs(epsg='4326')
force_clean_df.head()
locations_df = pd.DataFrame()
locationlist=[]
locations_df['LOCATION_LONGITUDE']=force_clean_df['LOCATION_LONGITUDE'].astype(float)
locations_df['LOCATION_LATITUDE'] =force_clean_df['LOCATION_LATITUDE'].astype(float)
for i, r in locations_df.iterrows():
    locationlist.append([r['LOCATION_LONGITUDE'],r['LOCATION_LATITUDE']])
fig1,ax = plt.subplots(1,2,figsize=(20,10))
police_df_Austin.plot(ax=ax[0],column='SECTOR',alpha=0.5,legend=True)
s=force_clean_df['INCIDENT_REASON']
s.value_counts().plot(kind='bar',ax=ax[1],rot=10)
force_clean_df.plot(marker='.',ax=ax[0])

force_clean_df.SUBJECT_RACE.value_counts()
print(force_clean_df.SUBJECT_RACE.value_counts())
fig1, ax1 = plt.subplots()
ax1.pie(force_clean_df.SUBJECT_RACE.value_counts(),labels=force_clean_df.SUBJECT_RACE.value_counts().keys(),autopct='%1.1f%%',startangle=90,  shadow=True)
ax1.axis('equal')
plt.show()

census_poverty_df = pd.read_csv('../input/data-science-for-good/cpe-data/'+
                                'Dept_37-00027/37-00027_ACS_data/37-00027_ACS_poverty/'+
                                'ACS_15_5YR_S1701_with_ann.csv')
census_poverty_df = census_poverty_df.iloc[1:].reset_index(drop=True)
census_poverty_df = census_poverty_df.rename(columns={'GEO.id2':'GEOID'})
census_tracts_gdf = gpd.read_file("../input/texgeo/cb_2017_48_tract_500k /cb_2017_48_tract_500k//cb_2017_48_tract_500k.shp")
census_merged_gdf = census_tracts_gdf.merge(census_poverty_df, on = 'GEOID')
census_merged_gdf = census_merged_gdf.to_crs(epsg='4326')
census_merged_gdf.head()
mapa = folium.Map([30.3, -97.7],zoom_start=10, height=500)
locations_df = force_clean_df[["LOCATION_LATITUDE", "LOCATION_LONGITUDE"]].copy()
locations_df = locations_df.iloc[locations_df[['LOCATION_LATITUDE','LOCATION_LONGITUDE']].dropna().index].reset_index(drop=True)
locations_df["LOCATION_LATITUDE"] = locations_df["LOCATION_LATITUDE"].astype('float')
locations_df["LOCATION_LONGITUDE"] = locations_df["LOCATION_LONGITUDE"].astype('float')
locationlist = locations_df.values.tolist()[-2000:]
for point in range(0, len(locationlist)):
    folium.CircleMarker(locationlist[point], radius=0.1, color='red').add_to(mapa)

mapa
DB_district_list=[k for k in force_clean_df['LOCATION_DISTRICT'].value_counts().keys()]
DB_district_list
final_aus_arrest=pd.read_csv('../input/aus-final/Dept_37-00027_arrest_GEo.csv')
final_aus_arrest.head()
force_df = pd.read_csv("../input/data-science-for-good/cpe-data/Dept_24-00013/24-00013_UOF_2008-2017_prepped.csv")
force_clean_df = force_df.loc[1:].reset_index(drop=True)
force_clean_df ['LOCATION_LONGITUDE']= pd.to_numeric(force_clean_df['LOCATION_LONGITUDE'], downcast='float') 
force_clean_df ['LOCATION_LATITUDE']= pd.to_numeric(force_clean_df['LOCATION_LATITUDE'], downcast='float') 
force_clean_df = force_clean_df[np.isfinite(force_clean_df['LOCATION_LONGITUDE'])]
force_clean_df=force_clean_df[force_clean_df['LOCATION_LONGITUDE']!=0].reset_index(drop=True)
foo=lambda x: Point(x['LOCATION_LONGITUDE'],x['LOCATION_LATITUDE'])
force_clean_df['geometry'] = (force_clean_df.apply(foo, axis=1))
force_clean_df = gpd.GeoDataFrame(force_clean_df, geometry='geometry')
force_clean_df.crs = {'init' :'epsg:4326'}
force_clean_df.head()
police_df = gpd.read_file( '../input/data-science-for-good/cpe-data//Dept_24-00013/'+
                '24-00013_Shapefiles/Minneapolis_Police_Precincts.shp')
police_df.head()
fig1,ax = plt.subplots(1,2,figsize=(20,10))
police_df.plot(ax=ax[0],alpha=0.5,legend=True)
s=force_clean_df['REASON_FOR_FORCE']
s.value_counts().plot(kind='bar',ax=ax[1])
force_clean_df.plot(marker='.',ax=ax[0],column='LOCATION_DISTRICT',legend=True)
force_clean_df.SUBJECT_RACE.value_counts()
print(force_clean_df.SUBJECT_RACE.value_counts())
fig1, ax1 = plt.subplots()
ax1.pie(force_clean_df.SUBJECT_RACE.value_counts(),labels=force_clean_df.SUBJECT_RACE.value_counts().keys(),autopct='%1.1f%%',startangle=90,  shadow=True)
ax1.axis('equal')
plt.show()
census_tract_df=gpd.read_file("../input/minneapolis/cb_2017_27_tract_500k /cb_2017_27_tract_500k.shp")
census_tract_df.head()
mapa = folium.Map([45, -93.3], height=500, zoom_start=11)

folium.GeoJson(police_df).add_to(mapa)
locations_df = force_clean_df[["LOCATION_LATITUDE", "LOCATION_LONGITUDE"]].copy()
notna = locations_df[['LOCATION_LATITUDE','LOCATION_LONGITUDE']].dropna().index
locations_df = locations_df.iloc[notna].reset_index(drop=True)
locations_df["LOCATION_LATITUDE"] = locations_df["LOCATION_LATITUDE"].astype('float')
locations_df["LOCATION_LONGITUDE"] = locations_df["LOCATION_LONGITUDE"].astype('float')
locationlist = locations_df.values.tolist()[-2000:]
for point in range(0, len(locationlist)):
    folium.CircleMarker(locationlist[point], radius=0.1, color='red').add_to(mapa)

mapa 
overlap_police=gpd.GeoDataFrame(columns=census_tract_df.columns)
item_set=[]
for index1,x in police_df.iterrows():
    lst_geoid=[]
    for index2, y in census_tract_df.iterrows():
        if x['geometry'].contains(y['geometry']) or y['geometry'].intersects(x['geometry']) or y['geometry'].contains(x['geometry']):
            if y['GEOID'] not in item_set:
                lst_geoid.append(y['GEOID'])
                item_set.append(y['GEOID'])
                police_df.at[index1,'GEOid']=lst_geoid
                overlap_police.loc[-1]=y
                overlap_police.index = overlap_police.index + 1
fig2,ax2 = plt.subplots()
force_clean_df.plot(ax=ax2,marker='.',column='LOCATION_DISTRICT',legend=True,markersize=20)
overlap_police.plot(ax=ax2,color='0.7',alpha=.5,edgecolor='white')

fig2.set_size_inches(10,10)
final_aus_arrest=pd.read_csv('../input/min-fin/Dept_2400013_arrest_GEo.csv')
final_aus_arrest.head()
force_df = pd.read_csv("../input/data-science-for-good/cpe-data/Dept_24-00098/24-00098_Vehicle-Stops-data.csv")
force_clean_df = force_df.loc[1:].reset_index(drop=True)
force_clean_df ['LOCATION_LONGITUDE']= pd.to_numeric(force_clean_df['LOCATION_LONGITUDE'], downcast='float') 
force_clean_df ['LOCATION_LATITUDE']= pd.to_numeric(force_clean_df['LOCATION_LATITUDE'], downcast='float') 
force_clean_df = force_clean_df[np.isfinite(force_clean_df['LOCATION_LONGITUDE'])]
force_clean_df=force_clean_df[force_clean_df['LOCATION_LONGITUDE']!=0].reset_index(drop=True)
foo=lambda x: Point(x['LOCATION_LONGITUDE'],x['LOCATION_LATITUDE'])
force_clean_df['geometry'] = (force_clean_df.apply(foo, axis=1))
force_clean_df = gpd.GeoDataFrame(force_clean_df, geometry='geometry')
force_clean_df.crs = {'init' :'epsg:4326'}
force_clean_df.head()
police_df = gpd.read_file('../input/data-science-for-good/cpe-data/Dept_24-00098/24-00098_Shapefiles/StPaul_geo_export_6646246d-0f26-48c5-a924-f5a99bb51c47.shp')
police_df.head()

fig2,ax2 = plt.subplots()
force_clean_df.plot(ax=ax2)
police_df.plot(ax=ax2,color='0.7',alpha=.5,edgecolor='white')

fig2.set_size_inches(10,10)
force_clean_df.SUBJECT_RACE.value_counts()
print(force_clean_df.SUBJECT_RACE.value_counts())
fig1, ax1 = plt.subplots()
ax1.pie(force_clean_df.SUBJECT_RACE.value_counts(),labels=force_clean_df.SUBJECT_RACE.value_counts().keys(),autopct='%1.1f%%',startangle=90,  shadow=True)
ax1.axis('equal')
plt.show()
census_tract_df=gpd.read_file("../input/stpaul/cb_2015_27_tract_500k/cb_2015_27_tract_500k.shp")
census_tract_df.plot()
final_arrest=pd.read_csv('../input/stpa-final/Dept_2400098_arrest_GEo.csv')
final_arrest.head()
force_df = pd.read_csv("../input/data-science-for-good/cpe-data/Dept_37-00049/37-00049_UOF-P_2016_prepped.csv")
force_clean_df = force_df.loc[1:].reset_index(drop=True)
force_clean_df ['LOCATION_LONGITUDE']= pd.to_numeric(force_clean_df['LOCATION_LONGITUDE'], downcast='float') 
force_clean_df ['LOCATION_LATITUDE']= pd.to_numeric(force_clean_df['LOCATION_LATITUDE'], downcast='float') 
force_clean_df = force_clean_df[np.isfinite(force_clean_df['LOCATION_LONGITUDE'])]
force_clean_df=force_clean_df[force_clean_df['LOCATION_LONGITUDE']!=0].reset_index(drop=True)
foo=lambda x: Point(x['LOCATION_LONGITUDE'],x['LOCATION_LATITUDE'])
force_clean_df['geometry'] = (force_clean_df.apply(foo, axis=1))
force_clean_df = gpd.GeoDataFrame(force_clean_df, geometry='geometry')
force_clean_df.crs = {'init' :'epsg:4326'}
force_clean_df.head()

police_df = gpd.read_file('../input/data-science-for-good/cpe-data/Dept_37-00049/37-00049_Shapefiles/EPIC.shp')
police_df=police_df.to_crs(epsg='4236')
police_df.head()
fig2,ax2 = plt.subplots()
force_clean_df.plot(ax=ax2)
police_df.plot(ax=ax2,color='0.7',alpha=.5,edgecolor='white')

fig2.set_size_inches(10,10)
force_clean_df.SUBJECT_RACE.value_counts()
print(force_clean_df.SUBJECT_RACE.value_counts())
fig1, ax1 = plt.subplots()
ax1.pie(force_clean_df.SUBJECT_RACE.value_counts(),labels=force_clean_df.SUBJECT_RACE.value_counts().keys(),autopct='%1.1f%%',startangle=90,  shadow=True)
ax1.axis('equal')
plt.show()
mapa = folium.Map([32.78, -96.79],zoom_start=10, height=500)
locations_df = force_clean_df[["LOCATION_LATITUDE", "LOCATION_LONGITUDE"]].copy()
locations_df = locations_df.iloc[locations_df[['LOCATION_LATITUDE','LOCATION_LONGITUDE']].dropna().index].reset_index(drop=True)
locations_df["LOCATION_LATITUDE"] = locations_df["LOCATION_LATITUDE"].astype('float')
locations_df["LOCATION_LONGITUDE"] = locations_df["LOCATION_LONGITUDE"].astype('float')
locationlist = locations_df.values.tolist()[-2000:]
for point in range(0, len(locationlist)):
    folium.CircleMarker(locationlist[point], radius=0.1, color='red').add_to(mapa)

mapa
census_tract_df=gpd.read_file("../input/dallas/cb_2017_48_tract_500k /cb_2017_48_tract_500k/cb_2017_48_tract_500k.shp")
census_tract_df.plot()
final_arrest=pd.read_csv('../input/dala-fin/Dept_3700049_arrest_GEo.csv')
final_arrest.head()
force_df = pd.read_csv("../input/data-science-for-good/cpe-data/Dept_35-00016/35-00016_UOF-OIS-P.csv")
force_clean_df = force_df.loc[1:].reset_index(drop=True)
force_clean_df ['LOCATION_LONGITUDE']= pd.to_numeric(force_clean_df['LOCATION_LONGITUDE'], downcast='float') 
force_clean_df ['LOCATION_LATITUDE']= pd.to_numeric(force_clean_df['LOCATION_LATITUDE'], downcast='float') 
force_clean_df = force_clean_df[np.isfinite(force_clean_df['LOCATION_LONGITUDE'])]
force_clean_df=force_clean_df[force_clean_df['LOCATION_LONGITUDE']!=0].reset_index(drop=True)
foo=lambda x: Point(x['LOCATION_LONGITUDE'],x['LOCATION_LATITUDE'])
force_clean_df['geometry'] = (force_clean_df.apply(foo, axis=1))
force_clean_df = gpd.GeoDataFrame(force_clean_df, geometry='geometry')
force_clean_df.crs = {'init' :'epsg:4326'}
force_clean_df.head()

police_df = gpd.read_file('../input/data-science-for-good/cpe-data/Dept_35-00016/35-00016_Shapefiles/OrlandoPoliceSectors.shp')
police_df=police_df.to_crs(epsg='4236')
police_df.head()
fig2,ax2 = plt.subplots()
force_clean_df.plot(ax=ax2)
police_df.plot(ax=ax2,color='0.7',alpha=.5,edgecolor='white')

fig2.set_size_inches(10,10)
force_clean_df.SUBJECT_RACE.value_counts()
print(force_clean_df.SUBJECT_RACE.value_counts())
fig1, ax1 = plt.subplots()
ax1.pie(force_clean_df.SUBJECT_RACE.value_counts(),labels=force_clean_df.SUBJECT_RACE.value_counts().keys(),autopct='%1.1f%%',startangle=90,  shadow=True)
ax1.axis('equal')
plt.show()
mapa = folium.Map([28.53, -81.39],zoom_start=10, height=500)
locations_df = force_clean_df[["LOCATION_LATITUDE", "LOCATION_LONGITUDE"]].copy()
locations_df = locations_df.iloc[locations_df[['LOCATION_LATITUDE','LOCATION_LONGITUDE']].dropna().index].reset_index(drop=True)
locations_df["LOCATION_LATITUDE"] = locations_df["LOCATION_LATITUDE"].astype('float')
locations_df["LOCATION_LONGITUDE"] = locations_df["LOCATION_LONGITUDE"].astype('float')
locationlist = locations_df.values.tolist()[-2000:]
for point in range(0, len(locationlist)):
    folium.CircleMarker(locationlist[point], radius=0.1, color='red').add_to(mapa)

mapa
census_tract_df=gpd.read_file("../input/orlando/cb_2016_12_tract_500k/cb_2016_12_tract_500k.shp")
census_tract_df.plot()
final_arrest=pd.read_csv('../input/orlan-final/Dept_35-00016_arrest_GEo.csv')
final_arrest.head()
