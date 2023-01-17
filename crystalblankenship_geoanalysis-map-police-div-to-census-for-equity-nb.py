import numpy as np
import pandas as pd 
import geopandas as geopd
import os
import folium
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

data_file_root = "../input/data-science-for-good/cpe-data"
census_data_root = '../input/census_tract_shapefile'

###SET DATA HOLDERS
shapefile_set = {'shapefile_data': {}, 'shapefile_crs': {}, 'shapefile_geom_type': {}}
ACS_set = {}
census_shapefile_set = {}

###SET MAPPINGS
city_coords = {'seattle': [47.6156301, -122.337831],
                'indaianapolis': [39.81, -86.26060805912148],
                'dallas': [],
                'boston': [],
                'charlotte': [],
                'austin': []}
shape_to_city = {'SPD_BEATS_WGS84.shp':'seattle',
                'Indianapolis_Police_Zones.shp':'indaianapolis',
                'EPIC.shp': 'dallas',
                'boston_police_districts_f55.shp': 'boston',
                'CMPD_Police_Division_Offices.shp': 'charlotte',
                'APD_DIST.shp': 'austin'}
dep_to_city = {'11-00091': 'boston',
                '23-00089': 'indaianapolis',
                '35-00103': 'charlotte',
                '37-00027': 'austin',
                '37-00049': 'dallas',
                '49-00009': 'seattle'}
city_to_census_state = {'seattle': '53',
                'indaianapolis': '18',
                'dallas': '48',
                'boston': '25',
                'charlotte': '37',
                'austin': '48'}
census_state_to_city = {'53': 'seattle',
                '18': 'indaianapolis',
                '48': 'dallas',
                '25': 'boston',
                '37': 'charlotte',
                '48': 'austin'}
city_utm_crs = {'seattle': {'init': 'epsg:32610'}, #UTM 10 for Seattle
                'indaianapolis': None,
                'dallas': None,
                'boston': None,
                'charlotte': None,
                'austin': None}

sqm_to_sqft = 10.7639
###GET DIRECTORIES FOR DIVISION DATA
directories = os.listdir(data_file_root)
directories = [d for d in directories if not d.endswith('.csv')]
print("Found shapefile directories: %s" % (tuple(directories),))

###LOOP THOURGH DIVISIONS AND GET SHAPEFILE DATA
for directory in directories:
    sub_directory = os.listdir(os.path.join(data_file_root, directory))
    shapefile_directories  = [f for f in sub_directory if "Shapefiles" in f]
    for s_directory in shapefile_directories:
        for shapefile in os.listdir(os.path.join(data_file_root, directory, s_directory)):
            if shapefile.endswith('.shp'):
                data_file = os.path.join(data_file_root, directory, s_directory, shapefile)
                data = geopd.read_file(data_file)
                shapefile_set['shapefile_data'][shapefile] = data
                shapefile_set['shapefile_crs'][shapefile] = data.crs
                shapefile_set['shapefile_geom_type'][shapefile] = data.geom_type.unique().tolist()
                #data.plot()
                
#GET SHAPEFILES THAT ARE POINTS BC CANNOT DO INTERSECTION AND AREA CALCS WITH POINTS!
point_shapefiles = [s for s,g in shapefile_set['shapefile_geom_type'].items() if 'Point' in g or 'Line' in g]
#GET SHAPEFILES WITH NO COORDINATE SYSTEM BC CANNOT DO INTERSECT WITHOUT CRS!
CRS_missing_shapefiles = [s for s,g in shapefile_set['shapefile_crs'].items() if not g]

#PRINT DATA FINDINGS
print("Found shapefiles: %s" % (tuple(shapefile_set['shapefile_data'].keys()),))
print("Coordinate systems of each shapefile for confirming crs exists: %s" % (shapefile_set['shapefile_crs'],))
#SHOW WARNINGS FOR SHAPEFILES THAT WILL NOT BE PROCESSED
if point_shapefiles:
    print("WARNING: Shapefiles with incorrect geometry found. Cannot porcess shapefiles: %s" % (point_shapefiles,))
if CRS_missing_shapefiles:
    print("WARNING: Shapefiles with missing CRS found. Cannot porcess shapefiles: %s" % (CRS_missing_shapefiles,))
###LOOP THROUGH DIVISIONS AND GET ACS DATA
for directory in directories:
    sub_directories = os.listdir(os.path.join(data_file_root, directory))
    ACS_directories  = [f for f in sub_directories if "ACS" in f]
    for ACS_directory in ACS_directories :
        if os.path.isdir(os.path.join(data_file_root, directory, ACS_directory)) == True:
            ACS_sub_directories = os.listdir(os.path.join(data_file_root, directory, ACS_directory))
            ACS_set[ACS_directory] = {}
            for ACS_sub_directory in ACS_sub_directories:
                if os.path.isdir(os.path.join(data_file_root, directory, ACS_directory, ACS_sub_directory)) == True:
                    ACS_data_type = ACS_sub_directory.split('_')[-1]
                    for ACS_file in os.listdir(os.path.join(data_file_root, directory, ACS_directory, ACS_sub_directory)):
                        if 'metadata' not in ACS_file:
                            data_file = os.path.join(data_file_root, directory, ACS_directory, ACS_sub_directory, ACS_file)
                            data = pd.read_csv(data_file)
                            if data.empty:
                                pass
                            else:
                                data.columns = data.iloc[0,:]
                                data=data.iloc[1:,:]
                            ACS_set[ACS_directory][ACS_data_type] = data
#PRINT EXAMPLE OF ACS DATA
print("ACS example data:")
print(ACS_set['49-00009_ACS_data']['race-sex-age'].iloc[0])
def make_map(shapefile, coords, city):
    map_map = folium.Map(coords, height=500, zoom_start=10)
    folium.GeoJson(shapefile).add_to(map_map)
    display(map_map)

city = shape_to_city['SPD_BEATS_WGS84.shp']
coords = city_coords['seattle']
data = shapefile_set['shapefile_data']['SPD_BEATS_WGS84.shp']
make_map(data, coords, city)
###loop thorugh census tracts and get shapefiles for each state. NEED CENSUS TRACTS FOR OBTAINING % CALCS OF CENSUS IN DIVISION.
census_dir_root = '../input/census-tract-shapefile'
census_directories = os.listdir(census_dir_root)
for c_sub_directory in census_directories:
    files = os.listdir(os.path.join(census_dir_root, c_sub_directory))
    for file in [f for f in files if f.endswith('.shp')]:
        state_ID = file.split('_')[2]
        shapefile = os.path.join(census_dir_root, c_sub_directory, file)
        data = geopd.read_file(shapefile)
        census_shapefile_set[state_ID] = data
#PRINT EXAMPLE OF CENSUS DATA  
print("census shapefiles loaded for: %s" % (tuple([census_state_to_city[s] for s in census_shapefile_set.keys()]),))
print("Census example data:")
print(census_shapefile_set[city_to_census_state['seattle']].iloc[0])
#reproject coordinate systems to match.
#change to same coordinate system for proper intersection. Mercator UTM-good for getting areas as well.
crs = {'init': 'epsg:32610'} #UTM 10 for Seattle
divisions = shapefile_set['shapefile_data']['SPD_BEATS_WGS84.shp'].to_crs(crs=crs)
census_tracts = census_shapefile_set[city_to_census_state['seattle']].to_crs(crs=crs)
#plot washington tracts
census_tracts.plot(color='blue', edgecolor='red')
plt.suptitle('Washington Tracts', fontsize=20)
txt = 'Tracts can be intersected with divisions and joined with acs data to do a manual spatial join of info per tract.'
plt.text(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=12)
plt.savefig('washington_tracts_ex-UTM 10.png')
#get area for whole tracts.
census_tracts['area_sqm_tract'] = census_tracts.geometry.area
census_tracts['area_sqft_tract'] = census_tracts.geometry.area*sqm_to_sqft
#get intersection of divisions to tracts
#THIS WILL CREATE ONE POLYGON FOR EACH PART THAT INTERSECTS EACH DIVISION. THIS WILL ALLOW US TO CALCULATE % 
#FOR CENSUS DATA AND THEN SUM BASED ON DIVISION IT FALLS WITHIN.
intersection = geopd.overlay(divisions, census_tracts, how='intersection')

#overlay intersections with divisions
fig, ax = plt.subplots()
ax.set_aspect('equal')
fig.suptitle('Seattle Division-Tracts Intersect', fontsize=20)
intersection.plot(ax=ax, color='blue', edgecolor='red', alpha=0.7)
divisions.plot(ax=ax, color='none', facecolor="none", edgecolor='black')
txt = 'The intersection polygons are cut based on each division outline.if tract crosses division it will be cut.'
fig.text(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=12)
plt.savefig('seattle_intersect_ex-UTM 10.png')
#lets's look at one division as an example of how intersection works, C2
fig, ax = plt.subplots()
ax.set_aspect('equal')
fig.suptitle('Seattle Intersect beat C2 Ex.', fontsize=20)
divisions.plot(ax=ax, color='none', facecolor="none", edgecolor='black')
intersection[intersection.beat == 'C2'].plot(ax=ax, color='blue', edgecolor='red', alpha=0.7)
txt = 'showing how tracts will be cut based on the division outline.'
fig.text(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=12)
plt.savefig('c2_intersect-UTM 10.png')
#get area for tract intersections/parts within each division
intersection['area_sqm_tract_inter'] = intersection.geometry.area
intersection['area_sqft_tract_inter'] = intersection.geometry.area*sqm_to_sqft
#calculate portion of tract areas within each division
intersection['tract_perc_in_div'] =  intersection['area_sqm_tract_inter']/intersection['area_sqm_tract'] 
print("Intersection with calculated column for percentage tract in division:")
print (intersection[['beat','GEOID','area_sqm_tract','area_sqm_tract_inter','tract_perc_in_div']].sort_values('beat').iloc[150])
#Join ACS data to intersections.
#Join ACS data based on GEOID.
ACS_set['49-00009_ACS_data']['race-sex-age']['GEOID'] = ACS_set['49-00009_ACS_data']['race-sex-age']['Id2']
intersection_merge = intersection.merge(ACS_set['49-00009_ACS_data']['race-sex-age'], on='GEOID')
#lets look at just the calcualtion for african americnas within each division.
intersection_merge[['beat','GEOID','Estimate; RACE - One race - Black or African American','tract_perc_in_div']]
#convert to float
intersection_merge['Estimate; RACE - One race - Black or African American'] = intersection_merge['Estimate; RACE - One race - Black or African American'].astype(float)
intersection_merge['tract_perc_in_div'] = intersection_merge['tract_perc_in_div'].astype(float)
#calculate % af. am. in tract division intersect
intersection_merge['afam_perct_tract'] = intersection_merge['Estimate; RACE - One race - Black or African American'] * intersection_merge['tract_perc_in_div']
#let's see exmaple pf tract-division calculation
example_calc_150 = intersection_merge[['afam_perct_tract', 'Estimate; RACE - One race - Black or African American', 'tract_perc_in_div', 'beat','GEOID']] .iloc[150]
print("Single tract example:")
print("%f percent of tract %s falls within beat %s." % (float(example_calc_150.tract_perc_in_div),example_calc_150.GEOID, example_calc_150.beat))
print("total af. am in tract is %f. %f percent of this is %f"% (example_calc_150['Estimate; RACE - One race - Black or African American'],
                                                        example_calc_150.tract_perc_in_div, example_calc_150.afam_perct_tract))

#lets now dissolve by division and aggregate based on new calculated column afam_perct_tract.
#clean up the dataframe first by limiting columns
intersection_for_dissolve = intersection_merge[['objectid','geometry','beat','GEOID','afam_perct_tract']]
#dissolve and sum the total af. am. in division based on our calculated column.
divisions_w_census_agg = intersection_for_dissolve.dissolve(by='beat', aggfunc='sum')

#FINALLY LETS SEE HOW MANY AF. AM. FALL WITHIN DIVISION.
fig, ax = plt.subplots()
ax.set_aspect('equal')
divisions_w_census_agg.plot(ax=ax, column = 'afam_perct_tract', scheme='equal_interval', cmap='YlOrRd', legend=True)
fig.suptitle('Seattle: African American in Each Division', fontsize=20)
lgd = ax.get_legend()
lgd.set_bbox_to_anchor((0.5, -0.1))
plt.savefig('Seattle_African_American_in_Each_Division.png', bbox_inches='tight')
#lets bring in incident data so we can compare num of incidents for certian demographics to demographics of the division itself.
seattle_incidents = pd.read_csv("../input/seattle-stops/terry-stops.csv")
seattle_incidents.head()