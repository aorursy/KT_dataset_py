import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.offline as py
py.init_notebook_mode(connected=True)
import seaborn as sns
from scipy.stats.mstats import gmean
import math
from scipy.stats.stats import pearsonr
import folium
from folium.plugins import MarkerCluster
from folium import IFrame
import shapely
from shapely.geometry import Point, Polygon
import unicodedata
import pysal as ps
import geopandas as gpd
from mpl_toolkits.basemap import Basemap
import geojson

%matplotlib inline

sns.set(rc={"figure.figsize": (20,10), "axes.titlesize" : 18, "axes.labelsize" : 12, 
            "xtick.labelsize" : 14, "ytick.labelsize" : 14 }, 
        palette=sns.color_palette("OrRd_d", 20))

import warnings
warnings.filterwarnings('ignore')

!cp ../input/images/kenya_admin3_level_nightlight_composite.png .
# Functions to read in and preprocess data

def preprocess_dhs_data(country, household_file, househole_member_file, births_file, cluster_file):
    # Load original DHS data 
    # The following error occurrs if we do not set convert_categoricals=False: ValueError: Categorical categories must be unique
    household_dhs_df = pd.read_stata('../input/'+country+'-dhs-household/'+household_file, convert_categoricals=False)
    household_member_dhs_df = pd.read_stata('../input/'+country+'-dhs-household-member/'+househole_member_file, convert_categoricals=False)
    births_dhs_df = pd.read_stata('../input/'+country+'-dhs-births/'+births_file, convert_categoricals=False)
    dhs_cluster_df = pd.read_csv('../input/'+country+'-dhs-cluster/'+cluster_file)

    # Keep only relevant features from each dataset
    household_dhs_df = household_dhs_df[['hv001', 'hv002', 'hv009', 'hv010',  'hv011',  'hv012',  'hv014',  
                                         'hv024',  'hv025', 'hv027',
                                         'hv206','hv201','hv204','hv205','hv225', 'hv226','hv213',
                                         'hv207', 'hv208', 'hv243a', 'hv221',
                                        'hv210', 'hv211', 'hv212', 'hv243c', 'hv243d',
                                         'hv209', 'hv244', 'hv245', 'hv246', 
                                         'hv247']]
    household_member_dhs_df = household_member_dhs_df[['hv001', 'hv002', 'hc31', 'hc70', 'hc73', 'hc2', 'hc3','ha1', 
                                                       'ha40', 'hv105', 'hv108', 'hv121']]
    births_dhs_df = births_dhs_df[['v001', 'v002',  'b2', 'b3', 'b5', 'b7']]

    # Save the resulting dataframes
    household_dhs_df.to_csv(country+'_household_dhs.csv', index = False)
    household_member_dhs_df.to_csv(country+'_household_member_dhs.csv', index = False)
    births_dhs_df.to_csv(country+'_births_dhs.csv', index = False)

    # DHS Cluster data preprocessing
    # drop irrelevant columns
    dhs_cluster_df.drop(columns=['GPS_Dataset', 'DHSCC', 'DHSYEAR', 'SurveyID'], inplace=True)
    dhs_cluster_df = dhs_cluster_df[dhs_cluster_df.columns.drop(list(dhs_cluster_df.filter(regex='1985')))]
    dhs_cluster_df = dhs_cluster_df[dhs_cluster_df.columns.drop(list(dhs_cluster_df.filter(regex='1990')))]
    dhs_cluster_df = dhs_cluster_df[dhs_cluster_df.columns.drop(list(dhs_cluster_df.filter(regex='1995')))]
    dhs_cluster_df = dhs_cluster_df[dhs_cluster_df.columns.drop(list(dhs_cluster_df.filter(regex='2000')))]
    dhs_cluster_df = dhs_cluster_df[dhs_cluster_df.columns.drop(list(dhs_cluster_df.filter(regex='2005')))]
    dhs_cluster_df = dhs_cluster_df[dhs_cluster_df.columns.drop(list(dhs_cluster_df.filter(regex='UN_Population')))]
    dhs_cluster_df = dhs_cluster_df[dhs_cluster_df.columns.drop(list(dhs_cluster_df.filter(regex='SMOD')))]
    dhs_cluster_df = dhs_cluster_df[dhs_cluster_df.columns.drop(list(dhs_cluster_df.filter(regex='Slope')))]
    dhs_cluster_df = dhs_cluster_df[dhs_cluster_df.columns.drop(list(dhs_cluster_df.filter(regex='Temperature')))]
    
    # Replace negative values with 0
    dhs_cluster_df['Nightlights_Composite'] = np.where(dhs_cluster_df['Nightlights_Composite']<0, 0, dhs_cluster_df['Nightlights_Composite'])
    
    dhs_cluster_df.to_csv(country+'_dhs_cluster.csv', index = False)
    
# States-Provinces shapefile
states_provinces_gdf = gpd.read_file('../input/world-geo-data/ne_10m_admin_1_states_provinces.shp')
# Kiva subnational MPI dataset
mpi_subnational_df = pd.read_csv('../input/kiva-mpi-subnational-with-coordinates/mpi_subnational_coords.csv')

# This step is just to ensure we have matches where possible between the two datasets
#from string import punctuation
states_provinces_gdf['name'] = states_provinces_gdf['name'].str.replace('-',' ')
mpi_subnational_df['Sub-national region'] = mpi_subnational_df['Sub-national region'].str.replace('-',' ')

def read_data(country, household_path, household_member_path, births_path, dhs_cluster_path, dhs_geo_path, 
              admin1_geo_path, admin2_geo_path):
    global household_dhs_df
    global household_member_dhs_df
    global births_dhs_df
    global dhs_cluster_df
    global dhs_geo_gdf
    global admin1_geo_gdf
    global admin2_geo_gdf
    
    # Read in preprocessed DHS datasets
    household_dhs_df = pd.read_csv(household_path)
    household_member_dhs_df = pd.read_csv(household_member_path)
    births_dhs_df = pd.read_csv(births_path)
    dhs_cluster_df = pd.read_csv(dhs_cluster_path)
    # DHS shapefile
    dhs_geo_gdf = gpd.read_file(dhs_geo_path)
    # Admin1 boundaries shapefile
    admin1_geo_gdf = gpd.read_file(admin1_geo_path)
    # Admin2 boundaries shapefile
    admin2_geo_gdf = gpd.read_file(admin2_geo_path)
# Functions to process DHS data at raw feature level

# Determine drinking water deprivation
clean_water_source = [10, 11, 12, 13, 20, 21, 30, 31, 41, 51, 71]
def determine_water_depriv(row):
    if row.hv201 in clean_water_source:
        if (row.hv204 != 996) & (row.hv204 >= 30):
            return 1
        else:
            return 0
    else:
        return 1

# Determine asset deprivation given information_asset, mobility_asset and livelihood_asset features
def determine_asset_depriv(row):
    if row.information_asset == 0:
        return 1
    if (row.mobility_asset == 0) & (row.livelihood_asset == 0):
        return 1
    return 0
    
def process_household_data(df):
    df.rename(columns={'hv009':'total_household_members'}, inplace=True)
    df['financial_depriv'] = np.where(df['hv247'] == 0, 1, 0) 
    df['electricity_depriv'] = np.where(df['hv206'] == 0, 1, 0)
    df['water_depriv'] = df.apply(determine_water_depriv, axis=1)
    improved_sanitation =  [10, 11, 12, 13, 14, 15, 21, 22, 41]
    df['sanitation_depriv'] = np.where((df.hv225 == 0) & (df['hv205'].isin(improved_sanitation)), 0, 1)
    df['cooking_fuel_depriv'] = np.where(df['hv226'].isin([6, 7, 8, 9, 10, 11, 95, 96]), 1, 0)
    df['floor_depriv'] = np.where(df['hv213'].isin([11, 12, 13, 96]), 1, 0) 
    df['information_asset'] =  np.where((df.hv207 == 1) | (df.hv208 == 1) | (df.hv243a == 1) | (df.hv221 == 1), 1, 0)
    df['mobility_asset'] =  np.where((df.hv210 == 1) | (df.hv211 == 1) | (df.hv212 == 1) | (df.hv243c == 1) | (df.hv243d == 1), 1, 0)
    df['livelihood_asset'] =  np.where((df.hv209 == 1) | (df.hv244 == 1) | (df.hv245 == 1) | (df.hv246 == 1), 1, 0)
    df['asset_depriv'] = df.apply(determine_asset_depriv, axis=1)
    return df

# Nutrition:
z_cutoff_malnutrition = -200 # Below -2 Std deviations is considered malnourished (UNDP documentation)
bmi_cutoff_malnutrition = 1850 # Cutoff according is 18.5 (UNDP documentation)

def process_malnutrition(row):
    if not math.isnan(row['hc31']):
        if (row['hv105'] < 5): # < 5 years old
            if(row['hc70'] <= z_cutoff_malnutrition): # use Ht/A Std deviations
                return 1
            else:
                return 0
    elif not math.isnan(row['ha1']):
        if (row['hv105'] >= 15) & (row['hv105'] <= 49) & (row['ha40'] <= bmi_cutoff_malnutrition): # use BMI for adults
            return 1
        else:
            return 0
    else:
        return np.nan
    
def process_household_member_data(df):
    df['malnutrition'] = df.apply(process_malnutrition, axis=1)
    df['child_not_in_school'] = np.where((df['hv105'] >= 7) & (df['hv105'] <= 14) & (df['hv121'] == 0), 1, 0)
    df['child_under_5'] = np.where(df['hv105'] < 5, 1, 0)
    df['woman_15_to_49'] = np.where((df['ha1'] >= 15) & (df['ha1'] <=49), 1, 0)
    aggregations = {
        'hv108':lambda x: x.ge(6).sum(), # count number in houseold with >= 6 years of school
        'malnutrition': 'sum',
        'child_under_5': 'max',
        'woman_15_to_49': 'max',
        'child_not_in_school': 'max'
    }
    summary_df = df.groupby(['hv001', 'hv002']).agg(aggregations).reset_index()
    summary_df['school_attainment_depriv'] = np.where(summary_df['hv108'] == 0, 1, 0)
    summary_df['school_attendance_depriv'] = np.where(summary_df['child_not_in_school'] == 0, 0, 1)
    return summary_df

five_year_threshold = 2009 # Since the survey year was 2014 
def child_mortailty(row):
    if (row.b5 == 0) & (row.b2+(row.b7/12) >= five_year_threshold):
        return 1
    else:
        return 0
    
def process_births_data(df):
    df['child_mortailty'] = df.apply(child_mortailty, axis=1)
    aggregations = {
        'child_mortailty': 'sum'
    }
    return df.groupby(['v001', 'v002']).agg(aggregations).reset_index()

def combine_datasets(household_df, household_member_df, births_df):
    print("Original DHS household dataset: ", household_df.shape)
    combined_df = household_df.merge(household_member_df)
    combined_df = combined_df.merge(births_df, how='left', left_on=['hv001', 'hv002'], right_on=['v001', 'v002'])
    print("Merged dataset: ", combined_df.shape)
    
    # drop irrelevant columns
    combined_df = combined_df[combined_df.columns.drop(list(combined_df.filter(regex='^hv2')))]
    combined_df = combined_df[combined_df.columns.drop(list(combined_df.filter(regex='^v0')))]
    return combined_df
# MPI Calculation function and function to filter out eligible households

def calculate_deprivations(df, dhs_cluster_df, mp_threshold):
    # Calculate headcount ratio and poverty intensity
    df['headcount_poor'] =  np.where(df['total_of_weighted_deprivations'] >= mp_threshold, df['total_household_members'], 0)
    df['total_poverty_intensity'] = df['headcount_poor']*df['total_of_weighted_deprivations']

    # Format the DHSID to get just the number part for matching with hv001
    dhs_cluster_df['DHSID_num'] = dhs_cluster_df['DHSID'].str[6:].str.lstrip('0').astype(int)
    
    # Merge dhs_cluster with dhs_geo
    print("Original dhs_cluster_df dataset: ", dhs_cluster_df.shape)
    dhs_cluster_df = dhs_cluster_df.merge(dhs_geo_gdf[['DHSID', 'ADM1NAME', 'LATNUM', 'LONGNUM']], left_on=['DHSID'], right_on=['DHSID'], suffixes=('', '_y'))
    dhs_cluster_df = dhs_cluster_df[dhs_cluster_df.columns.drop(list(dhs_cluster_df.filter(regex='_y')))]
    print("Merged dhs_cluster_df dataset: ", dhs_cluster_df.shape)

    # Merge combined_df with dhs_cluster data to get county information (name)
    df = df.merge(dhs_cluster_df[['DHSID_num', 'ADM1NAME', 'LATNUM', 'LONGNUM', 'Nightlights_Composite', 'All_Population_Density_2015', 
                                  'BUILT_Population_2014' ]], left_on=['hv001'], right_on=['DHSID_num'])
    print("Merged df dataset: ", df.shape)
    return df

# Aggregate to specifed level, COUNTY level by default
def aggregate_admin_level(df, level='ADM1NAME', col='mpi_county'):
    aggregations = {
        'headcount_poor': 'sum',
        'total_household_members': 'sum',
        'total_poverty_intensity': 'sum',
        'Nightlights_Composite': 'mean',
        'All_Population_Density_2015': 'mean',
        'BUILT_Population_2014': 'mean'
    }
    df = df.groupby([level]).agg(aggregations).reset_index()

    # Calculate MPI at the required aggregation level
    df['headcount_ratio'] = df['headcount_poor']/df['total_household_members']
    df['poverty_intensity'] = df['total_poverty_intensity']/df['headcount_poor']
    df[col] = df['headcount_ratio'] * df['poverty_intensity']
    
    # Calculate nightlight population density ratio (per 10000/km^2)
    df['nightlight_ratio'] = df['Nightlights_Composite']/(df['All_Population_Density_2015']/10000)
    return df

def get_combined_data_for_eligible_households():
    global household_dhs_df
    global household_member_dhs_df
    global births_dhs_df
    
    # Process DHS data to get individual indicators
    household_dhs_df = process_household_data(household_dhs_df)
    household_member_dhs_summary_df = process_household_member_data(household_member_dhs_df)
    births_dhs_summary_df = process_births_data(births_dhs_df)
    combined_df = combine_datasets(household_dhs_df, household_member_dhs_summary_df, births_dhs_summary_df)

    # remove households with missing indicators
    print("Combined DHS Dataset: ", combined_df.shape)
    combined_df.dropna(inplace=True)
    print("Dataset after removing households with missing indicators: ", combined_df.shape)

    # remove ineligible households
    eligible_df = combined_df[(combined_df['woman_15_to_49'] != 0) | (combined_df['child_under_5'] != 0)]
    print("Dataset after removing ineligible households: ", eligible_df.shape)
    return eligible_df

def calculate_total_of_weighted_depriv(row):
    edu_ind_weight = 1/6
    health_ind_weight = 1/6
    liv_ind_weight = 1/18
    return (row.school_attainment_depriv*edu_ind_weight) + (row.school_attendance_depriv*edu_ind_weight) + (row.malnutrition*health_ind_weight) + (row.child_mortailty*health_ind_weight) + (row.electricity_depriv*liv_ind_weight) + (row.water_depriv*liv_ind_weight) + (row.sanitation_depriv*liv_ind_weight) + (row.cooking_fuel_depriv*liv_ind_weight) + (row.floor_depriv*liv_ind_weight) + (row.asset_depriv*liv_ind_weight)

# Function to run the whole process
# Note: The lines where sjoin is used are commented out in order to run on Kaggle servers. The data has been preprocessed locally,
# and read in when running on Kaggle. To run full sjoin steps, simple uncomment the lines.
def calculate_mpi(country, admin1_geo, admin1_col, admin1_mpi_col, 
                  admin2_geo=gpd.GeoDataFrame(), admin2_col='', admin2_mpi_col='', 
                  admin3_geo=gpd.GeoDataFrame(), admin3_col='', admin3_mpi_col=''):
    global household_dhs_df
    global household_member_dhs_df
    global births_dhs_df
    global dhs_mpi_df
    # Create them in case they are not produced
    admin2_dhs_mpi_df = pd.DataFrame()
    admin3_dhs_mpi_df = pd.DataFrame()

    eligible_df = get_combined_data_for_eligible_households()

    # calclate total weighted deprivations
    eligible_df['total_of_weighted_deprivations'] = eligible_df.apply(calculate_total_of_weighted_depriv, axis=1)

    # calculate MPI. mp_threshold is 0.333 because this is the cutoff for being considered multi-dimensionally poor 
    # (poor in more than one dimension, since there are 3 dimensions, this is 1/3)
    dhs_mpi_df = calculate_deprivations(eligible_df, dhs_cluster_df, 0.333)

    # Spatially join to admin1 boundaries
    #dhs_mpi_gdf = convert_to_geodataframe_with_lat_long(dhs_mpi_df, 'LONGNUM', 'LATNUM')
    #dhs_mpi_joined_gdf = gpd.sjoin(dhs_mpi_gdf, admin1_geo, op='within')
    #print("Dataset spatially joined with admin level 1 geodata: ", dhs_mpi_joined_gdf.shape)   
    #dhs_mpi_joined_gdf.to_csv(country+'_dhs_mpi_admin1_sjoin_nightlight.csv', index = False)
    dhs_mpi_joined_gdf = pd.read_csv('../input/'+country.lower()+'-preprocessed/'+country+'_dhs_mpi_admin1_sjoin_nightlight.csv')
    
    # Aggregate to admin1 (Province) level
    admin1_dhs_mpi_df = aggregate_admin_level(dhs_mpi_joined_gdf, level=admin1_col, col=admin1_mpi_col)
    print("Dataset aggregated to admin level 1: ", admin1_dhs_mpi_df.shape)
    
    # Ensure we are using title case for names (this is inconsistent in some country's datasets)
    admin1_dhs_mpi_df[admin1_col] = admin1_dhs_mpi_df[admin1_col].str.title()
    
    if not admin2_geo.empty:
        # Spatially join to admin2 boundaries
        #dhs_mpi_joined_gdf = gpd.sjoin(dhs_mpi_gdf, admin2_geo, op='within')
        #print("Dataset spatially joined with admin level 2 geodata: ", dhs_mpi_joined_gdf.shape)
        #dhs_mpi_joined_gdf.to_csv(country+'_dhs_mpi_admin2_sjoin_nightlight.csv', index = False)
        dhs_mpi_joined_gdf = pd.read_csv('../input/'+country.lower()+'-preprocessed/'+country+'_dhs_mpi_admin2_sjoin_nightlight.csv')
    if admin2_col:
        # Aggregate to admin2 (County) level
        admin2_dhs_mpi_df = aggregate_admin_level(dhs_mpi_joined_gdf, level=admin2_col, col=admin2_mpi_col)
        print("Dataset aggregated to admin level 2: ", admin2_dhs_mpi_df.shape)
    
    if not admin3_geo.empty:
        # Spatially join to admin3 boundaries
        #dhs_mpi_joined_gdf = gpd.sjoin(dhs_mpi_gdf, admin3_geo, op='within')
        #print("Dataset spatially joined with admin level 3 geodata: ", dhs_mpi_joined_gdf.shape)
        #dhs_mpi_joined_gdf.to_csv(country+'_dhs_mpi_admin3_sjoin_nightlight.csv', index = False)
        dhs_mpi_joined_gdf = pd.read_csv('../input/'+country.lower()+'-preprocessed/'+country+'_dhs_mpi_admin3_sjoin_nightlight.csv')
    if admin3_col:
        # Aggregate to admin3 level
        admin3_dhs_mpi_df = aggregate_admin_level(dhs_mpi_joined_gdf, level=admin3_col, col=admin3_mpi_col)
        print("Dataset aggregated to admin level 3: ", admin3_dhs_mpi_df.shape)

    return dhs_mpi_joined_gdf, admin1_dhs_mpi_df, admin2_dhs_mpi_df, admin3_dhs_mpi_df
# Geometry and joining functions

# Function to combine MPI subnational scores with geometry
def get_mpi_subnational_gdf(mpi_subnational_df, states_provinces_gdf, country):
    # Keep just country data
    states_provinces_gdf = states_provinces_gdf[states_provinces_gdf['admin'] == country]
    mpi_subnational_df = mpi_subnational_df[mpi_subnational_df['Country'] == country]

    print("Country states_provinces_gdf dataset: ", states_provinces_gdf.shape)
    print("Country mpi_subnational_df dataset: ", mpi_subnational_df.shape)
    states_provinces_gdf.drop_duplicates(subset='woe_label', keep="last", inplace=True)
    print("Cleaned states_provinces_gdf dataset: ", states_provinces_gdf.shape)

    mpi_subnational_df = mpi_subnational_df[mpi_subnational_df['Country'] == country]
    mpi_subnational_df = mpi_subnational_df.merge(states_provinces_gdf, left_on='Sub-national region', right_on='name')
    print("Merged mpi_subnational_gdf dataset (with states_provinces_gdf): ", mpi_subnational_df.shape)
    return mpi_subnational_df

# Define some geo conversion functions
# Spatially join to counties
def convert_to_geodataframe_with_lat_long(df, lon, lat):
    df['geometry'] = df.apply(lambda row: Point(row[lon], row[lat]), axis=1)
    gdf = gpd.GeoDataFrame( df, geometry='geometry')
    gdf.crs = {"init":'epsg:4326'}
    return gdf

def convert_to_geodataframe_with_geometry(df, geometry):
    gdf = gpd.GeoDataFrame( df, geometry='geometry')
    gdf.crs = {"init":'epsg:4326'}
    return gdf

# Replace polygons with simple ones
def replace_geometry(gdf, gdf_simple_path):
    gdf_simple = gpd.read_file(gdf_simple_path)
    gdf['geometry'] = gdf_simple['geometry']
    
def get_geo_gdf(country):
    return states_provinces_gdf[states_provinces_gdf['geonunit'] == country]

def nightlight_color(feature, feature_col, threshold_scale):
    #colors = ['#a3a38f','#b8b87a','#cccc66','#e0e052','#ebeb47','#ffff33' ]
    colors = ['#333322','#666644','#999977','#cccc99','#e6e6aa','#ffffbb' ]
    for i, val in enumerate(threshold_scale):
        #print("threshold i:", i, val, " feature: ", feature['properties'][feature_col])
        if feature['properties'][feature_col] < val:
            return colors[i]
    return colors[-1]

def create_nightlight_map(geo_gdf, feature_col, lat, long, zoom, threshold_scale):
    geojson = geo_gdf.to_json()
    country_map = folium.Map([lat, long], zoom_start = zoom)
    folium.GeoJson(
        geojson,
        style_function=lambda feature: {
            'fillColor': nightlight_color(feature, feature_col, threshold_scale),
            'fillOpacity' : 0.7,
            'color' : 'black',
            'weight' : 1
            }
        ).add_to(country_map)
    return country_map
# Uncomment the line below to run pre-processing of original DHS files
#preprocess_dhs_data('kenya', 'KEHR71FL.DTA', 'KEPR71FL.DTA', 'KEBR71FL.DTA', 'KEGC71FL.csv')
read_data('kenya', 
          '../input/kenya-preprocessed/kenya_household_dhs.csv',
          '../input/kenya-preprocessed/kenya_household_member_dhs.csv',
          '../input/kenya-preprocessed/kenya_births_dhs.csv',
          '../input/kenya-preprocessed/kenya_dhs_cluster.csv',
          '../input/kenya-preprocessed/KEGE71FL.shp', 
          '../input/kenya-humdata-admin-geo/Kenya_admin_2014_WGS84.shp', 
          '../input/kenya-humdata-admin-geo/KEN_Adm2.shp')

# Replace polygons with simple ones
replace_geometry(admin1_geo_gdf, '../input/kenya-humdata-admin-geo/Kenya_admin_2014_WGS84_simple.shp')
replace_geometry(admin2_geo_gdf, '../input/kenya-humdata-admin-geo/KEN_Adm2_simple.shp')

# Run the initial MPI calc again so that we can do comparisons
kenya_raw_mpi_df, kenya_admin1_mpi_df, kenya_admin2_mpi_df, kenya_admin3_mpi_df = calculate_mpi('Kenya', admin1_geo_gdf, 'Province', 'mpi_admin1', 
        admin2_geo=admin2_geo_gdf, admin2_col='ADM1NAME', admin2_mpi_col='mpi_admin2', 
        admin3_geo=admin2_geo_gdf, admin3_col='Adm2Name', admin3_mpi_col='mpi_admin3')
kenya_admin1_mpi_df.Nightlights_Composite.describe()
kenya_nightlight_threshold_scale = [0, 0.25, 0.5, 0.75, 1, 5] 
kenya_geo_gdf = get_geo_gdf('Kenya')
kenya_geo_gdf = kenya_geo_gdf.merge(kenya_admin1_mpi_df[['Province', 'Nightlights_Composite']], how='left', left_on='name', right_on='Province')
create_nightlight_map(kenya_geo_gdf, 'Nightlights_Composite', 0.0236, 37.9062, 6, kenya_nightlight_threshold_scale)
# Correlation between Nightlight Composite and MPI
print("Correlation, p-value: ", pearsonr(kenya_admin1_mpi_df.loc[:, 'mpi_admin1'], kenya_admin1_mpi_df.loc[:, 'Nightlights_Composite']))
sns.regplot(x=kenya_admin1_mpi_df.mpi_admin1, y=kenya_admin1_mpi_df.Nightlights_Composite).set_title('Kenya Admin Level 1 MPI vs Nightlight Composite')
admin1_geo_gdf = admin1_geo_gdf.merge(kenya_admin2_mpi_df[['ADM1NAME', 'Nightlights_Composite']], how='left', left_on='COUNTY', right_on='ADM1NAME')

# Replace NaN values with 0
admin1_geo_gdf['Nightlights_Composite'] = np.where(admin1_geo_gdf['Nightlights_Composite'].isnull(), 0, admin1_geo_gdf['Nightlights_Composite'])

create_nightlight_map(admin1_geo_gdf, 'Nightlights_Composite', 0.0236, 37.9062, 6, kenya_nightlight_threshold_scale)
# Correlation between Nightlight Composite and MPI
print("Correlation, p-value: ", pearsonr(kenya_admin2_mpi_df.loc[:, 'mpi_admin2'], kenya_admin2_mpi_df.loc[:, 'Nightlights_Composite']))
sns.regplot(x=kenya_admin2_mpi_df.mpi_admin2, y=kenya_admin2_mpi_df.Nightlights_Composite).set_title('Kenya Admin Level 2 MPI vs Nightlight Composite')
admin2_geo_gdf = admin2_geo_gdf.merge(kenya_admin3_mpi_df[['Adm2Name', 'Nightlights_Composite']], how='left', left_on='Adm2Name', right_on='Adm2Name')

create_nightlight_map(admin2_geo_gdf, 'Nightlights_Composite', 0.0236, 37.9062, 6, kenya_nightlight_threshold_scale)
# Disregard the areas for which there is no MPI
kenya_admin3_mpi_df = kenya_admin3_mpi_df[np.isfinite(kenya_admin3_mpi_df['mpi_admin3'])]

# Correlation between Nightlight Composite and MPI
print("Correlation, p-value: ", pearsonr(kenya_admin3_mpi_df.loc[:, 'mpi_admin3'], kenya_admin3_mpi_df.loc[:, 'Nightlights_Composite']))
sns.regplot(x=kenya_admin3_mpi_df.mpi_admin3, y=kenya_admin3_mpi_df.Nightlights_Composite).set_title('Kenya Admin Level 3 MPI vs Nightlight Composite')
# Correlation between Nightlight composite and population density
print("Correlation, p-value: ", pearsonr(kenya_admin3_mpi_df.loc[:, 'All_Population_Density_2015'], kenya_admin3_mpi_df.loc[:, 'Nightlights_Composite']))
sns.regplot(x=kenya_admin3_mpi_df.All_Population_Density_2015, y=kenya_admin3_mpi_df.Nightlights_Composite).set_title('Kenya Admin Level 3 Population Density vs Nightlight Composite')
# Disregard 0 nightlight areas
kenya_admin3_mpi_df = kenya_admin3_mpi_df[kenya_admin3_mpi_df['nightlight_ratio'] > 0]

# Correlation between Nightlight ratio and MPI
print("Correlation, p-value: ", pearsonr(kenya_admin3_mpi_df.loc[:, 'mpi_admin3'], kenya_admin3_mpi_df.loc[:, 'nightlight_ratio']))
sns.regplot(x=kenya_admin3_mpi_df.mpi_admin3, y=kenya_admin3_mpi_df.nightlight_ratio).set_title('Kenya Admin Level 3 MPI vs Nightlight Ratio')
kenya_cluster_mpi_df = aggregate_admin_level(kenya_raw_mpi_df, level='DHSID_num', col='mpi_cluster')
# Disregard the areas for which there is no MPI
kenya_cluster_mpi_df = kenya_cluster_mpi_df[np.isfinite(kenya_cluster_mpi_df['mpi_cluster'])]

# Correlation between Nightlight Composite and MPI
print("Correlation, p-value: ", pearsonr(kenya_cluster_mpi_df.loc[:, 'mpi_cluster'], kenya_cluster_mpi_df.loc[:, 'Nightlights_Composite']))
sns.regplot(x=kenya_cluster_mpi_df.mpi_cluster, y=kenya_cluster_mpi_df.Nightlights_Composite).set_title('Kenya DHS Cluster MPI vs Nightlight Composite')
# Correlation between Nightlight composite and population density
print("Correlation, p-value: ", pearsonr(kenya_cluster_mpi_df.loc[:, 'All_Population_Density_2015'], kenya_cluster_mpi_df.loc[:, 'Nightlights_Composite']))
sns.regplot(x=kenya_cluster_mpi_df.All_Population_Density_2015, y=kenya_cluster_mpi_df.Nightlights_Composite).set_title('Kenya DHS Cluster Population Density vs Nightlight Composite')
# Disregard the areas for which there is no MPI or nightlight ratio
kenya_cluster_mpi_df = kenya_cluster_mpi_df[np.isfinite(kenya_cluster_mpi_df['mpi_cluster'])]
kenya_cluster_mpi_df = kenya_cluster_mpi_df[np.isfinite(kenya_cluster_mpi_df['nightlight_ratio'])]
# Disregard 0 nightlight areas
kenya_cluster_mpi_df = kenya_cluster_mpi_df[kenya_cluster_mpi_df['nightlight_ratio'] > 0]

# Correlation between Nightlight ratio and MPI
print("Correlation, p-value: ", pearsonr(kenya_cluster_mpi_df.loc[:, 'mpi_cluster'], kenya_cluster_mpi_df.loc[:, 'nightlight_ratio']))
sns.regplot(x=kenya_cluster_mpi_df.mpi_cluster, y=kenya_cluster_mpi_df.nightlight_ratio).set_title('Kenya DHS Cluster MPI vs Nightlight Ratio')
# Uncomment the line below to run pre-processing of original DHS files
#preprocess_dhs_data('zimbabwe', 'ZWHR71FL.DTA', 'ZWPR71FL.DTA', 'ZWBR71FL.DTA', 'ZWGC71FL.csv')
# Read in DHS and Geo Data
read_data('zimbabwe', 
          '../input/zimbabwe-preprocessed/zimbabwe_household_dhs.csv',
          '../input/zimbabwe-preprocessed/zimbabwe_household_member_dhs.csv',
          '../input/zimbabwe-preprocessed/zimbabwe_births_dhs.csv',
          '../input/zimbabwe-preprocessed/zimbabwe_dhs_cluster.csv',
          '../input/zimbabwe-preprocessed/ZWGE72FL.shp', 
          '../input/zimbabwe-humdata-admin-geo/zwe_polbnda_adm1_250k_cso.shp', 
          '../input/zimbabwe-humdata-admin-geo/zwe_polbnda_adm2_250k_cso.shp'
         )

# Simplify geometry. Seems to be necessary only for admin level 2 for Zimbabwe.
replace_geometry(admin2_geo_gdf, '../input/zimbabwe-humdata-admin-geo/zwe_polbnda_adm2_250k_cso_simple.shp')
raw_mpi_df, admin1_mpi_df, admin2_mpi_df, admin3_mpi_df = calculate_mpi('Zimbabwe', admin1_geo_gdf, 'ADM1NAME', 'mpi_admin1', 
    admin2_geo=admin2_geo_gdf, admin2_col='DIST_NM_LA', admin2_mpi_col='mpi_admin2')
zimbabwe_nightlight_threshold_scale = [0, 0.25, 0.5, 0.75, 1, 5] 
admin1_geo_gdf = admin1_geo_gdf.merge(admin1_mpi_df[['ADM1NAME', 'Nightlights_Composite']], how='left', left_on='PROVINCE', right_on='ADM1NAME')

create_nightlight_map(admin1_geo_gdf, 'Nightlights_Composite', -19.0154, 29.1549, 6, zimbabwe_nightlight_threshold_scale)
# Correlation between Nightlight Composite and MPI
print("Correlation, p-value: ", pearsonr(admin1_mpi_df.loc[:, 'mpi_admin1'], admin1_mpi_df.loc[:, 'Nightlights_Composite']))
sns.regplot(x=admin1_mpi_df.mpi_admin1, y=admin1_mpi_df.Nightlights_Composite).set_title('Zimbabwe Admin Level 1 MPI vs Nightlight Composite')
admin2_geo_gdf = admin2_geo_gdf.merge(admin2_mpi_df[['DIST_NM_LA', 'Nightlights_Composite']], how='left', left_on='DIST_NM_LA', right_on='DIST_NM_LA')

create_nightlight_map(admin2_geo_gdf, 'Nightlights_Composite', -19.0154, 29.1549, 6, zimbabwe_nightlight_threshold_scale)
# Disregard the areas for which there is no MPI
admin2_mpi_df = admin2_mpi_df[np.isfinite(admin2_mpi_df['mpi_admin2'])]

# Correlation between Nightlight Composite and MPI
print("Correlation, p-value: ", pearsonr(admin2_mpi_df.loc[:, 'mpi_admin2'], admin2_mpi_df.loc[:, 'Nightlights_Composite']))
sns.regplot(x=admin2_mpi_df.mpi_admin2, y=admin2_mpi_df.Nightlights_Composite).set_title('Zimbabwe Admin Level 2 MPI vs Nightlight Composite')
dhs_cluster_df.Nightlights_Composite.describe()
# Correlation between Nightlight composite and population density
print("Correlation, p-value: ", pearsonr(admin2_mpi_df.loc[:, 'All_Population_Density_2015'], admin2_mpi_df.loc[:, 'Nightlights_Composite']))
sns.regplot(x=admin2_mpi_df.All_Population_Density_2015, y=admin2_mpi_df.Nightlights_Composite).set_title('Zimbabwe Admin Level 2 Population Density vs Nightlight Composite')
# Disregard 0 nightlight areas
admin2_mpi_df = admin2_mpi_df[admin2_mpi_df['nightlight_ratio'] > 0]

# Correlation between Nightlight ratio and MPI
print("Correlation, p-value: ", pearsonr(admin2_mpi_df.loc[:, 'mpi_admin2'], admin2_mpi_df.loc[:, 'nightlight_ratio']))
sns.regplot(x=admin2_mpi_df.mpi_admin2, y=admin2_mpi_df.nightlight_ratio).set_title('Zimbabwe Admin Level 2 MPI vs Nightlight Ratio')
cluster_mpi_df = aggregate_admin_level(raw_mpi_df, level='DHSID_num', col='mpi_cluster')
# Disregard the areas for which there is no MPI
cluster_mpi_df = cluster_mpi_df[np.isfinite(cluster_mpi_df['mpi_cluster'])]

# Correlation between Nightlight Composite and MPI
print("Correlation, p-value: ", pearsonr(cluster_mpi_df.loc[:, 'mpi_cluster'], cluster_mpi_df.loc[:, 'Nightlights_Composite']))
sns.regplot(x=cluster_mpi_df.mpi_cluster, y=cluster_mpi_df.Nightlights_Composite).set_title('Zimbabwe DHS Cluster MPI vs Nightlight Composite')
# Correlation between Nightlight composite and population density
print("Correlation, p-value: ", pearsonr(cluster_mpi_df.loc[:, 'All_Population_Density_2015'], cluster_mpi_df.loc[:, 'Nightlights_Composite']))
sns.regplot(x=cluster_mpi_df.All_Population_Density_2015, y=cluster_mpi_df.Nightlights_Composite).set_title('Zimbabwe DHS Cluster Population Density vs Nightlight Composite')
# Disregard the areas for which there is no MPI or nightlight ratio
cluster_mpi_df = cluster_mpi_df[np.isfinite(cluster_mpi_df['mpi_cluster'])]
cluster_mpi_df = cluster_mpi_df[np.isfinite(cluster_mpi_df['nightlight_ratio'])]
# Disregard 0 nightlight areas
cluster_mpi_df = cluster_mpi_df[cluster_mpi_df['nightlight_ratio'] > 0]

# Correlation between Nightlight ratio and MPI
print("Correlation, p-value: ", pearsonr(cluster_mpi_df.loc[:, 'mpi_cluster'], cluster_mpi_df.loc[:, 'nightlight_ratio']))
sns.regplot(x=cluster_mpi_df.mpi_cluster, y=cluster_mpi_df.nightlight_ratio).set_title('Zimbabwe DHS Cluster MPI vs Nightlight Ratio')
# Uncomment the line below to run pre-processing of original DHS files
#preprocess_dhs_data('rwanda', 'RWHR70FL.DTA', 'RWPR70FL.DTA', 'RWBR70FL.DTA', 'RWGC71FL.csv')
# Read in DHS and Geo Data
read_data('rwanda', 
          '../input/rwanda-preprocessed/rwanda_household_dhs.csv',
          '../input/rwanda-preprocessed/rwanda_household_member_dhs.csv',
          '../input/rwanda-preprocessed/rwanda_births_dhs.csv',
          '../input/rwanda-preprocessed/rwanda_dhs_cluster.csv',
          '../input/rwanda-preprocessed/RWGE72FL.shp', 
          '../input/rwanda-humdata-admin-geo/RWA_Admin2_2006_NISR.shp', 
          '../input/rwanda-humdata-admin-geo/RWA_Admin3_2006_NISR.shp')

# Simplify geometry. Seems to be necessary only for admin level 2 for Zimbabwe.
replace_geometry(admin2_geo_gdf, '../input/rwanda-humdata-admin-geo/RWA_Admin3_2006_NISR_simple.shp')

# Doing some manual recoding to get matches 
states_provinces_gdf.name.replace('Southern', 'South', inplace=True)
states_provinces_gdf.name.replace('Northern', 'North', inplace=True)
states_provinces_gdf.name.replace('Eastern', 'East', inplace=True)
states_provinces_gdf.name.replace('Western', 'West', inplace=True)

admin1_geo_gdf.PROVINCE.replace('SOUTHERN PROVINCE', 'South', inplace=True)
admin1_geo_gdf.PROVINCE.replace('NORTHERN PROVINCE', 'North', inplace=True)
admin1_geo_gdf.PROVINCE.replace('EASTERN PROVINCE', 'East', inplace=True)
admin1_geo_gdf.PROVINCE.replace('WESTERN PROVINCE', 'West', inplace=True)
admin1_geo_gdf.PROVINCE.replace('TOWN OF KIGALI', 'Kigali City', inplace=True)
raw_mpi_df, admin1_mpi_df, admin2_mpi_df, admin3_mpi_df = calculate_mpi('Rwanda', admin1_geo_gdf, 'ADM1NAME', 'mpi_admin1',
                                    admin2_geo=admin2_geo_gdf, admin2_col='NOMDISTR', admin2_mpi_col='mpi_admin2')
rwanda_nightlight_threshold_scale = [0, 0.25, 0.5, 0.75, 1, 5] 
admin1_geo_gdf = admin1_geo_gdf.merge(admin1_mpi_df[['ADM1NAME', 'Nightlights_Composite']], how='left', left_on='PROVINCE', right_on='ADM1NAME')

# Replace NaN values with 0
admin1_geo_gdf['Nightlights_Composite'] = np.where(admin1_geo_gdf['Nightlights_Composite'].isnull(), 0, admin1_geo_gdf['Nightlights_Composite'])

create_nightlight_map(admin1_geo_gdf, 'Nightlights_Composite', -1.9403, 29.8739, 8, rwanda_nightlight_threshold_scale)
# Correlation between Nightlight Composite and MPI
print("Correlation, p-value: ", pearsonr(admin1_mpi_df.loc[:, 'mpi_admin1'], admin1_mpi_df.loc[:, 'Nightlights_Composite']))
sns.regplot(x=admin1_mpi_df.mpi_admin1, y=admin1_mpi_df.Nightlights_Composite).set_title('Rwanda Admin Level 1 MPI vs Nightlight Composite')
admin2_geo_gdf = admin2_geo_gdf.merge(admin2_mpi_df[['NOMDISTR', 'Nightlights_Composite']], how='left', left_on='NOMDISTR', right_on='NOMDISTR')

create_nightlight_map(admin2_geo_gdf, 'Nightlights_Composite', -1.9403, 29.8739, 8, rwanda_nightlight_threshold_scale)
# Correlation between Nightlight Composite and MPI
print("Correlation, p-value: ", pearsonr(admin2_mpi_df.loc[:, 'mpi_admin2'], admin2_mpi_df.loc[:, 'Nightlights_Composite']))
sns.regplot(x=admin2_mpi_df.mpi_admin2, y=admin2_mpi_df.Nightlights_Composite).set_title('Rwanda Admin Level 2 MPI vs Nightlight Composite')
# Correlation between Nightlight composite and population density
print("Correlation, p-value: ", pearsonr(admin2_mpi_df.loc[:, 'All_Population_Density_2015'], admin2_mpi_df.loc[:, 'Nightlights_Composite']))
sns.regplot(x=admin2_mpi_df.All_Population_Density_2015, y=admin2_mpi_df.Nightlights_Composite).set_title('Rwanda Admin Level 2 Population Density vs Nightlight Composite')
# Disregard 0 nightlight areas
admin2_mpi_df = admin2_mpi_df[admin2_mpi_df['nightlight_ratio'] > 0]

# Correlation between Nightlight ratio and MPI
print("Correlation, p-value: ", pearsonr(admin2_mpi_df.loc[:, 'mpi_admin2'], admin2_mpi_df.loc[:, 'nightlight_ratio']))
sns.regplot(x=admin2_mpi_df.mpi_admin2, y=admin2_mpi_df.nightlight_ratio).set_title('Rwanda Admin Level 2 MPI vs Nightlight Ratio')
cluster_mpi_df = aggregate_admin_level(raw_mpi_df, level='DHSID_num', col='mpi_cluster')
# Disregard the areas for which there is no MPI
cluster_mpi_df = cluster_mpi_df[np.isfinite(cluster_mpi_df['mpi_cluster'])]

# Correlation between Nightlight Composite and MPI
print("Correlation, p-value: ", pearsonr(cluster_mpi_df.loc[:, 'mpi_cluster'], cluster_mpi_df.loc[:, 'Nightlights_Composite']))
sns.regplot(x=cluster_mpi_df.mpi_cluster, y=cluster_mpi_df.Nightlights_Composite).set_title('DHS Cluster MPI vs Nightlight Composite')
# Correlation between Nightlight composite and population density
print("Correlation, p-value: ", pearsonr(cluster_mpi_df.loc[:, 'All_Population_Density_2015'], cluster_mpi_df.loc[:, 'Nightlights_Composite']))
sns.regplot(x=cluster_mpi_df.All_Population_Density_2015, y=cluster_mpi_df.Nightlights_Composite).set_title('DHS Cluster Population Density vs Nightlight Composite')
# Disregard the areas for which there is no MPI or nightlight ratio
cluster_mpi_df = cluster_mpi_df[np.isfinite(cluster_mpi_df['mpi_cluster'])]
cluster_mpi_df = cluster_mpi_df[np.isfinite(cluster_mpi_df['nightlight_ratio'])]
# Disregard 0 nightlight areas
cluster_mpi_df = cluster_mpi_df[cluster_mpi_df['nightlight_ratio'] > 0]

# Correlation between Nightlight ratio and MPI
print("Correlation, p-value: ", pearsonr(cluster_mpi_df.loc[:, 'mpi_cluster'], cluster_mpi_df.loc[:, 'nightlight_ratio']))
sns.regplot(x=cluster_mpi_df.mpi_cluster, y=cluster_mpi_df.nightlight_ratio).set_title('DHS Cluster MPI vs Nightlight Ratio')