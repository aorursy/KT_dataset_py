import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.offline as py
py.init_notebook_mode(connected=True)
import seaborn as sns
from scipy.stats.mstats import gmean
import math
from scipy.stats.stats import pearsonr
#from geopandas.tools import sjoin
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

!cp ../input/images/rwanda_mpi2_decomposed_radar.png .
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
    dhs_cluster_df.to_csv(country+'_dhs_cluster.csv', index = False)

# Uncomment the line below to run pre-processing of original DHS files
#preprocess_dhs_data('kenya', 'KEHR71FL.DTA', 'KEPR71FL.DTA', 'KEBR71FL.DTA', 'KEGC71FL.csv')

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
    df = df.merge(dhs_cluster_df[['DHSID_num', 'ADM1NAME', 'LATNUM', 'LONGNUM']], left_on=['hv001'], right_on=['DHSID_num'])
    print("Merged df dataset: ", df.shape)
    return df

# Aggregate to specifed level, COUNTY level by default
def aggregate_admin_level(df, level='ADM1NAME', col='mpi_county'):
    aggregations = {
        'headcount_poor': 'sum',
        'total_household_members': 'sum',
        'total_poverty_intensity': 'sum'
    }
    df = df.groupby([level]).agg(aggregations).reset_index()

    # Calculate MPI at the required aggregation level
    df['headcount_ratio'] = df['headcount_poor']/df['total_household_members']
    df['poverty_intensity'] = df['total_poverty_intensity']/df['headcount_poor']
    df[col] = df['headcount_ratio'] * df['poverty_intensity']
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
def calculate_mpi(country, admin1_geo, admin1_col, admin1_mpi_col, admin2_geo=gpd.GeoDataFrame(), admin2_col='', admin2_mpi_col='', admin3_geo=gpd.GeoDataFrame(), admin3_col='', admin3_mpi_col=''):
    global household_dhs_df
    global household_member_dhs_df
    global births_dhs_df
    global dhs_mpi_df
    # Create them in case they are not produced
    admin2_dhs_mpi_df = pd.DataFrame()
    admin3_dhs_mpi_df = pd.DataFrame()
    
    # delete after debugging
    global dhs_mpi_joined_gdf

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
    #dhs_mpi_joined_gdf.to_csv(country+'_dhs_mpi_admin1_sjoin.csv', index = False)
    dhs_mpi_joined_gdf = pd.read_csv('../input/'+country.lower()+'-preprocessed/'+country+'_dhs_mpi_admin1_sjoin.csv')
    
    # Aggregate to admin1 (Province) level
    admin1_dhs_mpi_df = aggregate_admin_level(dhs_mpi_joined_gdf, level=admin1_col, col=admin1_mpi_col)
    print("Dataset aggregated to admin level 1: ", admin1_dhs_mpi_df.shape)
    
    # Ensure we are using title case for names (this is inconsistent in some country's datasets)
    admin1_dhs_mpi_df[admin1_col] = admin1_dhs_mpi_df[admin1_col].str.title()
    
    if not admin2_geo.empty:
        # Spatially join to admin2 boundaries
        #dhs_mpi_joined_gdf = gpd.sjoin(dhs_mpi_gdf, admin2_geo, op='within')
        #print("Dataset spatially joined with admin level 2 geodata: ", dhs_mpi_joined_gdf.shape)
        #dhs_mpi_joined_gdf.to_csv(country+'_dhs_mpi_admin2_sjoin.csv', index = False)
        dhs_mpi_joined_gdf = pd.read_csv('../input/'+country.lower()+'-preprocessed/'+country+'_dhs_mpi_admin2_sjoin.csv')
    if admin2_col:
        # Aggregate to admin2 (County) level
        admin2_dhs_mpi_df = aggregate_admin_level(dhs_mpi_joined_gdf, level=admin2_col, col=admin2_mpi_col)
        print("Dataset aggregated to admin level 2: ", admin2_dhs_mpi_df.shape)
    
    if not admin3_geo.empty:
        # Spatially join to admin3 boundaries
        #dhs_mpi_joined_gdf = gpd.sjoin(dhs_mpi_gdf, admin3_geo, op='within')
        #print("Dataset spatially joined with admin level 3 geodata: ", dhs_mpi_joined_gdf.shape)
        #dhs_mpi_joined_gdf.to_csv(country+'_dhs_mpi_admin3_sjoin.csv', index = False)
        dhs_mpi_joined_gdf = pd.read_csv('../input/'+country.lower()+'-preprocessed/'+country+'_dhs_mpi_admin3_sjoin.csv')
    if admin3_col:
        # Aggregate to admin3 level
        admin3_dhs_mpi_df = aggregate_admin_level(dhs_mpi_joined_gdf, level=admin3_col, col=admin3_mpi_col)
        print("Dataset aggregated to admin level 3: ", admin3_dhs_mpi_df.shape)

    return admin1_dhs_mpi_df, admin2_dhs_mpi_df, admin3_dhs_mpi_df
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

def create_map(geo_gdf, data, key_on, key_col, feature, fill_color, lat, long, zoom, threshold_scale):
    geojson = geo_gdf.to_json()
    country_map = folium.Map([lat, long], zoom_start = zoom)
    country_map.choropleth(
        geo_data=geojson,
        name=feature+' choropleth',
        key_on=key_on,
        fill_color=fill_color,
        data=data,
        columns=[key_col, feature],
        threshold_scale=threshold_scale,
        legend_name= feature+' per Province'
    )
    return country_map
edu_ind_weight_2 = 1/8     # 1/6 * 3/4
health_ind_weight_2 = 1/8  # 1/6 * 3/4
liv_ind_weight_2 = 1/24    # 1/18 * 3/4
fin_ind_weight_2 = 1/4
 
def calculate_total_of_weighted_depriv_2(row):
    return (row.school_attainment_depriv*edu_ind_weight_2) + (row.school_attendance_depriv*edu_ind_weight_2) + (row.malnutrition*health_ind_weight_2) + (row.child_mortailty*health_ind_weight_2) + (row.electricity_depriv*liv_ind_weight_2) + (row.water_depriv*liv_ind_weight_2) + (row.sanitation_depriv*liv_ind_weight_2) + (row.cooking_fuel_depriv*liv_ind_weight_2) + (row.floor_depriv*liv_ind_weight_2) + (row.asset_depriv*liv_ind_weight_2) + (row.financial_depriv*fin_ind_weight_2)
def calculate_mpi_2(country, admin1_geo, admin1_col, admin1_mpi2_col, admin2_geo=gpd.GeoDataFrame(), admin2_col='', admin2_mpi2_col=''):
    global household_dhs_df
    global household_member_dhs_df
    global births_dhs_df
    global dhs_mpi2_df
    
    eligible_df = get_combined_data_for_eligible_households()
    
    # calclate total weighted deprivations
    eligible_df['total_of_weighted_deprivations'] = eligible_df.apply(calculate_total_of_weighted_depriv_2, axis=1)

    # calculate MPI
    dhs_mpi2_df = calculate_deprivations(eligible_df, dhs_cluster_df, 0.25)

    # Spatially join to admin1 boundaries
    #dhs_mpi2_gdf = convert_to_geodataframe_with_lat_long(dhs_mpi2_df, 'LONGNUM', 'LATNUM')
    #dhs_admin1_mpi2_gdf = gpd.sjoin(dhs_mpi2_gdf, admin1_geo, op='within')
    #print("Dataset spatially joined with admin level 1 geodata: ", dhs_admin1_mpi2_gdf.shape)   
    #dhs_admin1_mpi2_gdf.to_csv(country+'_dhs_mpi2_admin1_sjoin.csv', index = False)
    dhs_admin1_mpi2_gdf = pd.read_csv('../input/'+country.lower()+'-preprocessed/'+country+'_dhs_mpi2_admin1_sjoin.csv')
    
    # Aggregate to admin1 (Province) level
    admin1_dhs_mpi2_df = aggregate_admin_level(dhs_admin1_mpi2_gdf, level=admin1_col, col=admin1_mpi2_col)
    print("Dataset aggregated to admin level 1: ", admin1_dhs_mpi2_df.shape)
    
    # Ensure we are using title case for names (this is inconsistent in some country's datasets)
    admin1_dhs_mpi2_df[admin1_col] = admin1_dhs_mpi2_df[admin1_col].str.title()
    
    if not admin2_geo.empty:
        # Spatially join to admin2 boundaries
        #dhs_admin2_mpi2_gdf = gpd.sjoin(dhs_mpi2_gdf, admin2_geo, op='within')
        #print("Dataset spatially joined with admin level 2 geodata: ", dhs_admin2_mpi2_gdf.shape)
        #dhs_admin2_mpi2_gdf.to_csv(country+'_dhs_mpi2_admin2_sjoin.csv', index = False)
        dhs_admin2_mpi2_gdf = pd.read_csv('../input/'+country.lower()+'-preprocessed/'+country+'_dhs_mpi2_admin2_sjoin.csv')
    if admin2_col:
        # Aggregate to admin2 (County) level
        admin2_dhs_mpi2_df = aggregate_admin_level(dhs_admin2_mpi2_gdf, level=admin2_col, col=admin2_mpi2_col)
        print("Dataset aggregated to admin level 2: ", admin2_dhs_mpi2_df.shape)

    return admin1_dhs_mpi2_df, admin2_dhs_mpi2_df, dhs_admin1_mpi2_gdf, dhs_admin2_mpi2_gdf
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

# Run the initial MPI calc again so that we can do comparisons
kenya_admin1_mpi_df, kenya_admin2_mpi_df, kenya_admin3_mpi_df = calculate_mpi('Kenya', admin1_geo_gdf, 'Province', 'mpi_admin1', 
        admin2_geo=admin2_geo_gdf, admin2_col='ADM1NAME', admin2_mpi_col='mpi_admin2', 
        admin3_col='Adm2Name', admin3_mpi_col='mpi_admin3')
# Merge 
kenya_mpi_subnational_gdf = get_mpi_subnational_gdf(mpi_subnational_df, states_provinces_gdf, 'Kenya')
kenya_admin1_mpi_merged_df = kenya_admin1_mpi_df.merge(kenya_mpi_subnational_gdf[['Sub-national region', 'MPI Regional']],
                                                left_on=['Province'], right_on=['Sub-national region'])
print("Dataset after merge with OPHI MPI data: ", kenya_admin1_mpi_merged_df.shape)
# Run MPI2 calc
kenya_admin1_mpi2_df, kenya_admin2_mpi2_df, kenya_admin1_mpi2_gdf, kenya_admin2_mpi2_gdf = calculate_mpi_2('Kenya', admin1_geo_gdf, 'Province', 'mpi2_admin1', 
        admin2_geo=admin2_geo_gdf, admin2_col='ADM1NAME', admin2_mpi2_col='mpi2_admin2')
kenya_mpi_threshold_scale = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6] # Define MPI scale for Kenya
kenya_geo_gdf = get_geo_gdf('Kenya')
create_map(kenya_geo_gdf, kenya_admin1_mpi2_df, 'feature.properties.name', 'Province', 'mpi2_admin1', 'YlOrRd', 0.0236, 37.9062, 6, kenya_mpi_threshold_scale)
# Merge 
country_mpi2_subnational_gdf = get_mpi_subnational_gdf(mpi_subnational_df, states_provinces_gdf, 'Kenya')
kenya_admin1_mpi2_merged_df = kenya_admin1_mpi2_df.merge(country_mpi2_subnational_gdf[['Sub-national region', 'MPI Regional']],
                                                left_on=['Province'], right_on=['Sub-national region'])
print("Dataset after merge with OPHI MPI data: ", kenya_admin1_mpi2_merged_df.shape)
    
# Check Correlation at admin1 level
print("Correlation, p-value: ", pearsonr(kenya_admin1_mpi2_merged_df.loc[:, 'mpi2_admin1'], kenya_admin1_mpi2_merged_df.loc[:, 'MPI Regional']))
sns.regplot(x="MPI Regional", y='mpi2_admin1', data=kenya_admin1_mpi2_merged_df)
# Aggregate to province
def aggregate_individual_indicators(df, region_col):
    aggregations = {
        'headcount_poor': 'sum',
        'total_household_members': 'sum',
        'total_poverty_intensity': 'sum',
        'electricity_depriv': 'sum',
        'water_depriv': 'sum',
        'sanitation_depriv': 'sum',
        'cooking_fuel_depriv': 'sum',
        'floor_depriv': 'sum',
        'asset_depriv': 'sum',
        'malnutrition': 'sum',
        'child_mortailty': 'sum',
        'school_attainment_depriv': 'sum',
        'school_attendance_depriv': 'sum',
        'financial_depriv' : 'sum'
    }
    return df.groupby([region_col]).agg(aggregations).reset_index()
    
province_dhs_mpi2_decomp_df = aggregate_individual_indicators(kenya_admin1_mpi2_gdf, 'Province' )

# Calculate deprivation raw headcount ratios
def get_headcount_ratios_for_all_indicators(df):
    df['electricity_depriv_ratio'] = df['electricity_depriv']/df['total_household_members']
    df['water_depriv_ratio'] = df['water_depriv']/df['total_household_members']
    df['sanitation_depriv_ratio'] = df['sanitation_depriv']/df['total_household_members']
    df['cooking_fuel_depriv_ratio'] = df['cooking_fuel_depriv']/df['total_household_members']
    df['floor_depriv_ratio'] = df['floor_depriv']/df['total_household_members']
    df['asset_depriv_ratio'] = df['asset_depriv']/df['total_household_members']
    df['malnutrition_ratio'] = df['malnutrition']/df['total_household_members']
    df['child_mortailty'] = df['child_mortailty']/df['total_household_members']
    df['school_attainment_depriv_ratio'] = df['school_attainment_depriv']/df['total_household_members']
    df['school_attendance_depriv_ratio'] = df['school_attendance_depriv']/df['total_household_members']
    df['financial_depriv_ratio'] = df['financial_depriv']/df['total_household_members']

    df['headcount_ratio'] = df['headcount_poor']/df['total_household_members']
    df['poverty_intensity'] = df['total_poverty_intensity']/df['headcount_poor']
    df['mpi'] = df['headcount_ratio'] * df['poverty_intensity']
    return df

province_mpi2_decomp_df = get_headcount_ratios_for_all_indicators(province_dhs_mpi2_decomp_df)
def radar_plot(df, columns, labels, title_col, num_rows, num_cols):
    fig = plt.figure(figsize=(5*num_cols,5*num_rows))
    for i, (name, row) in enumerate(df.iterrows()):
        stats=df.loc[i, columns].values

        # Create a color palette:
        palette = plt.cm.get_cmap("Set2", num_rows*num_cols)

        angles=np.linspace(0, 2*np.pi, len(columns), endpoint=False)
        # close the plot
        stats=np.concatenate((stats,[stats[0]]))
        angles=np.concatenate((angles,[angles[0]]))

        ax = plt.subplot(num_rows, num_cols, i+1, polar=True)
        ax.plot(angles, stats, linewidth=2, linestyle='solid', color=palette(i))
        ax.fill(angles, stats, color=palette(i), alpha=0.6)
        ax.set_theta_offset(1.85) # Change the rotation so that labels don't overlap
        ax.set_rmax(0.2)
        ax.set_rticks([0.05, 0.1, 0.15, 0.2])  # less radial ticks
        ax.set_thetagrids(angles * 180/np.pi, labels)
        ax.set_title(row[title_col], color=palette(i))
        ax.grid(True)
        
    plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=2.0, rect=[0, 0, 0.95, 0.95])
    plt.show()
columns=np.array(['electricity_depriv_ratio', 'water_depriv_ratio', 'sanitation_depriv_ratio', 'cooking_fuel_depriv_ratio', 'floor_depriv_ratio',
                 'asset_depriv_ratio','malnutrition_ratio','child_mortailty','school_attainment_depriv_ratio','school_attendance_depriv_ratio', 
                 'financial_depriv_ratio'])
labels=np.array(['elect', 'water', 'sanit', 'cook', 'flr',
                 'asset','maln','mort','sch1','sch2', 'fin'])
radar_plot(province_mpi2_decomp_df, columns, labels, 'Province', 3, 3)
# Correlation between calculated MPI1 and MPI2
print("Correlation, p-value: ", pearsonr(kenya_admin2_mpi2_df.loc[:, 'mpi2_admin2'], kenya_admin2_mpi_df.loc[:, 'mpi_admin2']))
sns.regplot(x=kenya_admin2_mpi_df.mpi_admin2, y=kenya_admin2_mpi2_df.mpi2_admin2)
create_map(admin1_geo_gdf, kenya_admin2_mpi2_df, 'feature.properties.COUNTY', 'ADM1NAME', 'mpi2_admin2', 'YlOrRd', 0.0236, 37.9062, 6, kenya_mpi_threshold_scale)
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

admin2_geo_gdf['NOMDISTR'] = admin2_geo_gdf['NOMDISTR'].str.title()

# Run the initial MPI calc again so that we can do comparisons
rwanda_admin1_mpi_df, rwanda_admin2_mpi_df, rwanda_admin3_mpi_df = calculate_mpi('Rwanda', admin1_geo_gdf, 'ADM1NAME', 'mpi_admin1', 
                                     admin2_geo=admin2_geo_gdf, admin2_col='NOMDISTR', admin2_mpi_col='mpi_admin2')
# Run MPI2 calc
rwanda_admin1_mpi2_df, rwanda_admin2_mpi2_df, rwanda_admin1_mpi2_gdf, rwanda_admin2_mpi2_gdf = calculate_mpi_2('Rwanda', admin1_geo_gdf, 'ADM1NAME', 'mpi2_admin1', 
                                         admin2_geo=admin2_geo_gdf, admin2_col='NOMDISTR', admin2_mpi2_col='mpi2_admin2')
rwanda_mpi_threshold_scale = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3] # Define MPI scale
rwanda_geo_gdf = get_geo_gdf('Rwanda')
create_map(rwanda_geo_gdf, rwanda_admin1_mpi2_df, 'feature.properties.name', 'ADM1NAME', 'mpi2_admin1', 'YlOrRd', -1.9403, 29.8739, 8, rwanda_mpi_threshold_scale)
# Merge 
country_mpi2_subnational_gdf = get_mpi_subnational_gdf(mpi_subnational_df, states_provinces_gdf, 'Rwanda')
rwanda_admin1_mpi2_merged_df = rwanda_admin1_mpi2_df.merge(country_mpi2_subnational_gdf[['Sub-national region', 'MPI Regional']],
                                                left_on=['ADM1NAME'], right_on=['Sub-national region'])
print("Dataset after merge with OPHI MPI data: ", rwanda_admin1_mpi2_merged_df.shape)
    
# Check Correlation at admin1 level
print("Correlation, p-value: ", pearsonr(rwanda_admin1_mpi2_merged_df.loc[:, 'mpi2_admin1'], rwanda_admin1_mpi2_merged_df.loc[:, 'MPI Regional']))
sns.regplot(x="MPI Regional", y='mpi2_admin1', data=rwanda_admin1_mpi2_merged_df)
province_mpi2_decomp_df = aggregate_individual_indicators(rwanda_admin1_mpi2_gdf, 'ADM1NAME' )
province_mpi2_decomp_df = get_headcount_ratios_for_all_indicators(province_mpi2_decomp_df)

columns=np.array(['electricity_depriv_ratio', 'water_depriv_ratio', 'sanitation_depriv_ratio', 'cooking_fuel_depriv_ratio', 'floor_depriv_ratio',
                 'asset_depriv_ratio','malnutrition_ratio','child_mortailty','school_attainment_depriv_ratio','school_attendance_depriv_ratio', 
                 'financial_depriv_ratio'])
labels=np.array(['elect', 'water', 'sanit', 'cook', 'flr',
                 'asset','maln','mort','sch1','sch2', 'fin'])
radar_plot(province_mpi2_decomp_df, columns, labels, 'ADM1NAME', 2, 3)
# Correlation between calculated MPI1 and MPI2
print("Correlation, p-value: ", pearsonr(rwanda_admin2_mpi2_df.loc[:, 'mpi2_admin2'], rwanda_admin2_mpi_df.loc[:, 'mpi_admin2']))
sns.regplot(x=rwanda_admin2_mpi_df.mpi_admin2, y=rwanda_admin2_mpi2_df.mpi2_admin2)
create_map(admin2_geo_gdf, rwanda_admin2_mpi2_df, 'feature.properties.NOMDISTR', 'NOMDISTR', 'mpi2_admin2', 'YlOrRd', -1.9403, 29.8739, 8, rwanda_mpi_threshold_scale)
def preprocess_intermed_data(country, filepath):
    df = pd.read_csv(filepath, encoding='cp1252')
    df = df[['AA1','AA2','AA3','AA4', 'Stratum', 
             #'Latitude', 
               'age', 'DG8a', 'DG8b', 'DG8c', 'poverty','ppi_score','UR','AA7', 
               'access_phone_SIM', 'own_phone_SIM', 'registered_mm', 'nonregistered_mm',
               'registered_bank_full', 'nonregistered_bank_full', 
               'FB22_1','FB22_2','FB22_3','FB22_4', 'FB3']]
    df.rename(columns={'FB22_1':'bank_savings'}, inplace=True)
    df.rename(columns={'FB22_2':'mm_savings'}, inplace=True)
    df.rename(columns={'FB22_3':'other_reg_savings'}, inplace=True)
    df.rename(columns={'FB22_4':'micro_savings'}, inplace=True)
    df.rename(columns={'FB3':'couldnt_access_credit'}, inplace=True)
    df['total_household_members'] = df['DG8a']+df['DG8b']+df['DG8c']
    df.to_csv(country+'_fii_preprocessed.csv', index = False)

def process_fii_data(df):
    df['acct_depriv'] = np.where((df['registered_bank_full']==1) | (df['registered_mm']==1), 0, 1)
    df['saving_depriv'] = df[['bank_savings','mm_savings','other_reg_savings','micro_savings']].min(axis=1)
    df['saving_depriv'].replace(1, 0, inplace=True)
    df['saving_depriv'].replace(2, 1, inplace=True)
    df['borrowing_depriv'] = np.where(df['couldnt_access_credit']==1, 1, 0)
    # Calculate financial deprivation indicator
    # Attempting to keep the definition uniform, lets say that someone is financially deprived if they are deprived in more than one
    # financial indicator.
    df['financial_depriv'] = np.where(df['acct_depriv'] + df['saving_depriv'] + df['borrowing_depriv'] > 1, 1, 0)
    return df

def calculate_total_weighted_fin_depriv(row):
    fin_ind_weight = 1/3
    return (row.acct_depriv*fin_ind_weight) + (row.saving_depriv*fin_ind_weight) + (row.borrowing_depriv*fin_ind_weight) 

def calculate_fin_deprivations(df):
    fin_ind_weight = 1/3
    mp_threshold = 1/3

    # calclate total weighted deprivations
    df['total_of_weighted_deprivations'] = df.apply(calculate_total_weighted_fin_depriv, axis=1)
    # Calculate headcount poor and poverty intensity
    df['headcount_poor'] =  np.where(df['total_of_weighted_deprivations'] >= mp_threshold, df['total_household_members'], 0)
    df['total_poverty_intensity'] = df['headcount_poor']*df['total_of_weighted_deprivations']
    return df

def calculate_mpi_with_fin_dimension(df, mpi_col, fin_poverty_col):
    mpi_weight = 3/4
    fin_weight = 1/4
    df['mpi2'] = (df[mpi_col]*mpi_weight) + (df[fin_poverty_col]*fin_weight)
    return df
    
def calculate_mpi2_improved_fin_dim(mpi_df, fin_df, mpi_region_col, fin_region_col, mpi_col):
    fin_df = process_fii_data(fin_df)
    fin_df = calculate_fin_deprivations(fin_df)
    fin_summary_df = aggregate_admin_level(fin_df, level=fin_region_col, col='fin_poverty')
    print("Dataset mpi_df: ", mpi_df.shape)
    mpi_fin_df = mpi_df.merge(fin_summary_df[[fin_region_col, 'fin_poverty']], how='left', left_on=[mpi_region_col], right_on=[fin_region_col])
    print("Dataset mpi_df after merge with fin_df: ", mpi_fin_df.shape)
    mpi_fin_df = calculate_mpi_with_fin_dimension(mpi_fin_df, mpi_col, 'fin_poverty')
    return mpi_fin_df

def check_correlation(mpi_fin_df, mpi_col, fin_poverty_col, mpi2_col):
    # Check Correlation at region level
    print("MPI vs Fin Poverty correlation, p-value: ", pearsonr(mpi_fin_df.loc[:, fin_poverty_col], mpi_fin_df.loc[:, mpi_col]))
    sns.regplot(x=mpi_col, y=fin_poverty_col, data=mpi_fin_df)
    plt.figure()
    print("MPI vs MPI2 correlation, p-value: ", pearsonr(mpi_fin_df.loc[:, mpi2_col], mpi_fin_df.loc[:, mpi_col]))
    sns.regplot(x=mpi_col, y=mpi2_col, data=mpi_fin_df)
    plt.figure()
#preprocess_intermed_data('kenya', '../input/financial-inclusion-insights/FII_ 2016_Kenya_Wave_4_Data.csv')
kenya_fin_df = pd.read_csv('../input/kenya-preprocessed/kenya_fii_preprocessed.csv',)

# Kenya-specific string processing
kenya_fin_df['Region'] = kenya_fin_df.Stratum.str.split('_').str[0]
kenya_fin_df['Region'] = kenya_fin_df['Region'].str.title()

# TODO: Update Region strings to match ADM1NAME
kenya_fin_df['Region'].replace('Muranga', "Murang'a", inplace=True)
kenya_fin_df['Region'].replace('Tharaka', "Tharaka-Nithi", inplace=True)

kenya_fin_df.sample(5)
kenya_mpi_fin_df = calculate_mpi2_improved_fin_dim(kenya_admin2_mpi_df, kenya_fin_df, 'ADM1NAME', 'Region', 'mpi_admin2')
kenya_admin1_geo_gdf = gpd.read_file('../input/kenya-humdata-admin-geo/Kenya_admin_2014_WGS84.shp')
replace_geometry(kenya_admin1_geo_gdf, '../input/kenya-humdata-admin-geo/Kenya_admin_2014_WGS84_simple.shp')
create_map(kenya_admin1_geo_gdf, kenya_mpi_fin_df, 'feature.properties.COUNTY', 'ADM1NAME', 'mpi2', 'YlOrRd', 0.0236, 37.9062, 6, kenya_mpi_threshold_scale)
check_correlation(kenya_mpi_fin_df, 'mpi_admin2', 'fin_poverty', 'mpi2')
plt.subplot(221).set_title("Kenya County MPI distribuion")
sns.distplot(kenya_mpi_fin_df.mpi_admin2, bins=30)

plt.subplot(222).set_title("Kenya County fin_poverty distribuion")
sns.distplot(kenya_mpi_fin_df.fin_poverty, bins=30)

plt.subplot(223).set_title("Kenya County MPI2 distribuion")
sns.distplot(kenya_mpi_fin_df.mpi2, bins=30)

plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=2.0, rect=[0, 0, 0.95, 0.95])
kenya_mpi_fin_df[['mpi_admin2', 'mpi2']].describe()
read_data('tanzania', 
          '../input/tanzania-preprocessed/tanzania_household_dhs.csv',
          '../input/tanzania-preprocessed/tanzania_household_member_dhs.csv',
          '../input/tanzania-preprocessed/tanzania_births_dhs.csv',
          '../input/tanzania-preprocessed/tanzania_dhs_cluster.csv',
          '../input/tanzania-preprocessed/TZGE7AFL.shp', 
          '../input/tanzania-humdata-admin-geo/tza_popa_adm1_regions_TNBS2012_OCHA.shp', 
          '../input/tanzania-humdata-admin-geo/tza_popa_adm2_districts_TNBS2012_OCHA.shp')
admin1_geo_gdf.sample()
# Run the initial MPI calc again so that we can do comparisons
tanzania_admin1_mpi_df, tanzania_admin2_mpi_df, tanzania_admin3_mpi_df = calculate_mpi('Tanzania', admin1_geo_gdf, 'REGION', 'mpi_admin1', 
        admin2_geo=admin2_geo_gdf, admin2_col='ADM1NAME', admin2_mpi_col='mpi_admin2')
# Run MPI2 calc
tanzania_admin1_mpi2_df, tanzania_admin2_mpi2_df, tanzania_admin1_mpi2_gdf, tanzania_admin2_mpi2_gdf = calculate_mpi_2('Tanzania', admin1_geo_gdf, 'REGION', 'mpi2_admin1', 
        admin2_geo=admin2_geo_gdf, admin2_col='ADM1NAME', admin2_mpi2_col='mpi2_admin2')
#preprocess_intermed_data('tanzania', '../input/financial-inclusion-insights/FII_2016_Tanzania_Wave_4_Data.csv')
tanzania_fin_df = pd.read_csv('../input/tanzania-preprocessed/tanzania_fii_preprocessed.csv')

# country-specific string processing
tanzania_fin_df['Stratum'] = tanzania_fin_df.Stratum.str.split('_').str[0]
tanzania_fin_df['Stratum'] = tanzania_fin_df['Stratum'].str.title()

# TODO: Update Region strings to match ADM1NAME
#kenya_fin_df['Region'].replace('Muranga', "Murang'a", inplace=True)
#kenya_fin_df['Region'].replace('Tharaka', "Tharaka-Nithi", inplace=True)

tanzania_fin_df.sample(5)
tanzania_mpi_fin_df = calculate_mpi2_improved_fin_dim(tanzania_admin2_mpi_df, tanzania_fin_df, 'ADM1NAME', 'Stratum', 'mpi_admin2')
# Drop the 2 districts with null fin_poverty and mpi2 (TODO: Investigate why they were null)
tanzania_mpi_fin_df.dropna(axis=0, how='any', inplace=True)
check_correlation(tanzania_mpi_fin_df, 'mpi_admin2', 'fin_poverty', 'mpi2')
plt.subplot(221).set_title("Tanzania County MPI distribuion")
sns.distplot(tanzania_mpi_fin_df.mpi_admin2, bins=30)

plt.subplot(222).set_title("Tanzania County fin_poverty distribuion")
sns.distplot(tanzania_mpi_fin_df.fin_poverty, bins=30)

plt.subplot(223).set_title("Tanzania County MPI2 distribuion")
sns.distplot(tanzania_mpi_fin_df.mpi2, bins=30)

plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=2.0, rect=[0, 0, 0.95, 0.95])
tanzania_mpi_fin_df[['mpi_admin2', 'mpi2']].describe()
def preprocess_wb_intermed_data(country, filepath):
    df = pd.read_csv(filepath)
    df.columns = map(str.lower, df.columns) # Convert column headers to lowercase for consistency
    df = df[['aa1', 'aa2', 'aa3', 'ur', 'dl26_1', 
             'own_phone', 'own_sim', 
             'fl13_1','fl13_2','fl13_3',
             'fl10_1','fl10_2','fl10_3','fl10_4',
             'registered_mm', 'registered_bank_full', 'registered_nbfi']]

    df.rename(columns={'aa1':'zone'}, inplace=True)
    df.rename(columns={'aa2':'district'}, inplace=True)
    df.rename(columns={'aa3':'county'}, inplace=True)
    df.rename(columns={'dl26_1':'total_household_members'}, inplace=True)
    
    df.rename(columns={'fl13_1':'bank_savings'}, inplace=True)
    df.rename(columns={'fl13_2':'mm_savings'}, inplace=True)
    df.rename(columns={'fl13_3':'other_reg_savings'}, inplace=True)

    df.rename(columns={'fl10_1':'bank_borrowing'}, inplace=True)
    df.rename(columns={'fl10_2':'micro_borrowing'}, inplace=True)
    df.rename(columns={'fl10_3':'mm_borrowing'}, inplace=True)
    df.rename(columns={'fl10_4':'other_formal_borrowing'}, inplace=True)
                
    df.to_csv(country+'_fii_preprocessed.csv', index = False)

def process_wb_fii_data(df):
    df['acct_depriv'] = np.where((df['registered_bank_full']==1) | (df['registered_mm']==1), 0, 1)
    df['saving_depriv'] = df[['bank_savings','mm_savings','other_reg_savings']].min(axis=1)
    df['saving_depriv'].replace(1, 0, inplace=True)
    df['saving_depriv'].replace(2, 1, inplace=True)
    
    df['bank_borrowing'].fillna(2, inplace=True)
    df['micro_borrowing'].fillna(2, inplace=True)
    df['mm_borrowing'].fillna(2, inplace=True)
    df['other_formal_borrowing'].fillna(2, inplace=True)
    
    df['borrowing_depriv'] = df[['bank_borrowing','micro_borrowing','mm_borrowing', 'other_formal_borrowing']].min(axis=1)
    df['borrowing_depriv'].replace(1, 0, inplace=True)
    df['borrowing_depriv'].replace(2, 1, inplace=True)
    # Calculate financial deprivation indicator
    # Attempting to keep the definition uniform, lets say that someone is financially deprived if they are deprived in more than one
    # financial indicator.
    df['financial_depriv'] = np.where(df['acct_depriv'] + df['saving_depriv'] + df['borrowing_depriv'] > 1, 1, 0)
    return df

def calculate_mpi2_improved_fin_dim(mpi_df, fin_df, mpi_region_col, fin_region_col, mpi_col):
    fin_df = process_wb_fii_data(fin_df)
    fin_df = calculate_fin_deprivations(fin_df)
    fin_summary_df = aggregate_admin_level(fin_df, level=fin_region_col, col='fin_poverty')
    print("Dataset mpi_df: ", mpi_df.shape)
    mpi_fin_df = mpi_df.merge(fin_summary_df[[fin_region_col, 'fin_poverty']], how='left', left_on=[mpi_region_col], right_on=[fin_region_col])
    print("Dataset mpi_df after merge with fin_df: ", mpi_fin_df.shape)
    mpi_fin_df = calculate_mpi_with_fin_dimension(mpi_fin_df, mpi_col, 'fin_poverty')
    return mpi_fin_df

def check_correlation(mpi_fin_df, mpi_col, fin_poverty_col, mpi2_col):
    # Check Correlation at region level
    print("MPI vs Fin Poverty correlation, p-value: ", pearsonr(mpi_fin_df.loc[:, fin_poverty_col], mpi_fin_df.loc[:, mpi_col]))
    sns.regplot(x=mpi_col, y=fin_poverty_col, data=mpi_fin_df)
    plt.figure()
    print("MPI vs MPI2 correlation, p-value: ", pearsonr(mpi_fin_df.loc[:, mpi2_col], mpi_fin_df.loc[:, mpi_col]))
    sns.regplot(x=mpi_col, y=mpi2_col, data=mpi_fin_df)
    plt.figure()
#preprocess_wb_intermed_data('rwanda', '../input/financial-inclusion-insights/final_cgap_rwanda_im_v2.csv')
rwanda_fin_df = pd.read_csv('../input/rwanda-preprocessed/rwanda_fii_preprocessed.csv')
# Replace region codes with names
fii_zone_map_df = pd.read_csv('../input/rwanda-preprocessed/rwanda_fii_zone_mappings.csv')
rwanda_fin_df['zone'].replace(dict(fii_zone_map_df.values), inplace=True)

fii_district_map_df = pd.read_csv('../input/rwanda-preprocessed/rwanda_fii_district_mappings.csv')
rwanda_fin_df['district'].replace(dict(fii_district_map_df.values), inplace=True)

fii_county_map_df = pd.read_csv('../input/rwanda-preprocessed/rwanda_fii_county_mappings.csv')
rwanda_fin_df['county'].replace(dict(fii_county_map_df.values), inplace=True)

rwanda_fin_df.sample(5)
rwanda_fin_df['zone'].replace('Kigali', 'Kigali City', inplace=True)
rwanda_mpi_fin_df = calculate_mpi2_improved_fin_dim(rwanda_admin1_mpi_df, rwanda_fin_df, 'ADM1NAME', 'zone', 'mpi_admin1')
check_correlation(rwanda_mpi_fin_df, 'mpi_admin1', 'fin_poverty', 'mpi2')
rwanda_mpi_fin_df = calculate_mpi2_improved_fin_dim(rwanda_admin2_mpi_df, rwanda_fin_df, 'NOMDISTR', 'district', 'mpi_admin2')
# Drop the 2 districts with null fin_poverty and mpi2 (TODO: Investigate why they were null)
rwanda_mpi_fin_df.dropna(axis=0, how='any', inplace=True)
check_correlation(rwanda_mpi_fin_df, 'mpi_admin2', 'fin_poverty', 'mpi2')
plt.subplot(221).set_title("Rwanda County MPI distribuion")
sns.distplot(rwanda_mpi_fin_df.mpi_admin2, bins=30)

plt.subplot(222).set_title("Rwanda County fin_poverty distribuion")
sns.distplot(rwanda_mpi_fin_df.fin_poverty, bins=30)

plt.subplot(223).set_title("Rwanda County MPI2 distribuion")
sns.distplot(rwanda_mpi_fin_df.mpi2, bins=30)

plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=2.0, rect=[0, 0, 0.95, 0.95])
rwanda_mpi_fin_df[['mpi_admin2', 'mpi2']].describe()
# Read in DHS and Geo Data
read_data('ghana', 
          '../input/ghana-preprocessed/ghana_household_dhs.csv',
          '../input/ghana-preprocessed/ghana_household_member_dhs.csv',
          '../input/ghana-preprocessed/ghana_births_dhs.csv',
          '../input/ghana-preprocessed/ghana_dhs_cluster.csv',
          '../input/ghana-preprocessed/GHGE71FL.shp', 
          '../input/ghana-humdata-admin-geo/GHA_admbndp1_1m_GAUL.shp', 
          '../input/ghana-humdata-admin-geo/GHA_admbndp2_1m_GAUL.shp')

# Simplify geometry.     
replace_geometry(admin2_geo_gdf, '../input/ghana-humdata-admin-geo/GHA_admbndp2_1m_GAUL_simple.shp')

ghana_admin1_mpi_df, ghana_admin2_mpi_df, ghana_admin3_mpi_df = calculate_mpi('Ghana', admin1_geo_gdf, 'ADM1_NAME', 'mpi_admin1', admin2_geo=admin2_geo_gdf, admin2_col='ADM2_NAME', admin2_mpi_col='mpi_admin2')
#preprocess_wb_intermed_data('ghana', '../input/financial-inclusion-insights/final_cgap_ghana_im_v2.csv')
ghana_fin_df = pd.read_csv('../input/ghana-preprocessed/ghana_fii_preprocessed.csv')
# Replace region codes with names
fii_zone_map_df = pd.read_csv('../input/ghana-preprocessed/ghana_fii_region_mappings.csv')
ghana_fin_df['zone'].replace(dict(fii_zone_map_df.values), inplace=True)
ghana_mpi_fin_df = calculate_mpi2_improved_fin_dim(ghana_admin1_mpi_df, ghana_fin_df, 'ADM1_NAME', 'zone', 'mpi_admin1')
check_correlation(ghana_mpi_fin_df, 'mpi_admin1', 'fin_poverty', 'mpi2')
plt.subplot(221).set_title("Ghana Region MPI distribuion")
sns.distplot(ghana_mpi_fin_df.mpi_admin1, bins=30)

plt.subplot(222).set_title("Ghana Region fin_poverty distribuion")
sns.distplot(ghana_mpi_fin_df.fin_poverty, bins=30)

plt.subplot(223).set_title("Ghana Region MPI2 distribuion")
sns.distplot(ghana_mpi_fin_df.mpi2, bins=30)

plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=2.0, rect=[0, 0, 0.95, 0.95])
ghana_mpi_fin_df[['mpi_admin1', 'mpi2']].describe()