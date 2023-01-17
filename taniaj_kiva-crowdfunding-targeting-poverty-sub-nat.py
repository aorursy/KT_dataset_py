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
from sklearn import metrics
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

!cp ../input/images/kenya_county_mpi_loans.png .
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
# Determine drinking water deprivation
# hv201 - source of drinking water
# hv204 - time to water and back (in minutes) 
# *Slightly different logic used compared to ophi calc but should work out the same
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
# A household is not deprived in assets if it has at least one asset from group (1) and at least one asset from groups (2) or (3).
def determine_asset_depriv(row):
    if row.information_asset == 0:
        return 1
    if (row.mobility_asset == 0) & (row.livelihood_asset == 0):
        return 1
    return 0
    
def process_household_data(df):
    # hv009 : total members in household, rename col
    df.rename(columns={'hv009':'total_household_members'}, inplace=True)
    
    # hv247 - bank account (used only in MPI2 model)
    df['financial_depriv'] = np.where(df['hv247'] == 0, 1, 0) 
        
    # hv206 - electricity, map to electricity_depriv
    df['electricity_depriv'] = np.where(df['hv206'] == 0, 1, 0)
    # hv201, hv 204 - map to water_depriv
    df['water_depriv'] = df.apply(determine_water_depriv, axis=1)
    # hv05 - type of toilet facility
    # hv25 - shared sanitation 
    # *Including 14, 15 as improved sanitation, ophi calculation does not.
    improved_sanitation =  [10, 11, 12, 13, 14, 15, 21, 22, 41]
    df['sanitation_depriv'] = np.where((df.hv225 == 0) & (df['hv205'].isin(improved_sanitation)), 0, 1)
    # hv26 - map to cooking_fuel_depriv
    df['cooking_fuel_depriv'] = np.where(df['hv226'].isin([6, 7, 8, 9, 10, 11, 95, 96]), 1, 0)
    # hv213 - floor type, map to floor_depriv
    df['floor_depriv'] = np.where(df['hv213'].isin([11, 12, 13, 96]), 1, 0)
    
    # hv207 Radio, HV208 - Television, hv243a - Mobile telephone, hv221 - Telephone (non-mobile)
    # hv210 Bicycle, HV211 Motorcycle or Scooter, HV212 Car or Truck, HV243C : Animal-drawn cart, HV243D : Boat with a motor
    # hv209 - Refrigerator, HV244 : Own land usable for agriculture, HV245 : Hectares for agricultural land
    # hv246 - Livestock, herds or farm animals
    # hv246a - cattle, hv246c - horses, hv246d - goats, hv246e - sheep, hv246f - chickens
    # * Note: I have used the simplified hv246 instead of individual livestock, slight deviation from OPHI calculation.
    df['information_asset'] =  np.where((df.hv207 == 1) | (df.hv208 == 1) | (df.hv243a == 1) | (df.hv221 == 1), 1, 0)
    df['mobility_asset'] =  np.where((df.hv210 == 1) | (df.hv211 == 1) | (df.hv212 == 1) | (df.hv243c == 1) | (df.hv243d == 1), 1, 0)
    df['livelihood_asset'] =  np.where((df.hv209 == 1) | (df.hv244 == 1) | (df.hv245 == 1) | (df.hv246 == 1), 1, 0)
    # determine asset_depriv
    df['asset_depriv'] = df.apply(determine_asset_depriv, axis=1)
    return df
# Nutrition:
# hc73: The measures are presented with two implied decimal places (no decimal points are included in the data file). 
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
    # determine if household member is malnourished
    df['malnutrition'] = df.apply(process_malnutrition, axis=1)
    
    # Education
    # Entrance age of primary: 6 years (http://stats.uis.unesco.org/unesco/TableViewer/tableView.aspx?ReportId=163)
    # hv121 - Household member attended school during current school year
    # hv105 - age
    df['child_not_in_school'] = np.where((df['hv105'] >= 7) & (df['hv105'] <= 14) & (df['hv121'] == 0), 1, 0)
    
    # Whether there is a child under 5 in a household or a woman between 15 and 49 are features required later on 
    # to determine whether a household is eligible for inclusion.
    df['child_under_5'] = np.where(df['hv105'] < 5, 1, 0)
    df['woman_15_to_49'] = np.where((df['ha1'] >= 15) & (df['ha1'] <=49), 1, 0)
    
    # Note: number of years of school is obtained slightly differently to the OPHI method.
    # Get summary stats per household
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
# V206 - Total number of sons who have died (children)
# V207 - Total number of daughters who have died  (children)
# b7 - Age at death of the child in completed months g (children)
# B2 Year of birth of child
# B5 Whether child was alive or dead at the time of interview. 
# Child mortality: a child has died in the household within the five years prior to the survey
five_year_threshold = 2009 # Since the survey year was 2014 

def child_mortailty(row):
    if (row.b5 == 0) & (row.b2+(row.b7/12) >= five_year_threshold):
        return 1
    else:
        return 0
    
def process_births_data(df):
    df['child_mortailty'] = df.apply(child_mortailty, axis=1)
    
    # Get summary stats per household
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
edu_ind_weight = 1/6
health_ind_weight = 1/6
liv_ind_weight = 1/18

def calculate_total_of_weighted_depriv(row):
    return (row.school_attainment_depriv*edu_ind_weight) + (row.school_attendance_depriv*edu_ind_weight) + (row.malnutrition*health_ind_weight) + (row.child_mortailty*health_ind_weight) + (row.electricity_depriv*liv_ind_weight) + (row.water_depriv*liv_ind_weight) + (row.sanitation_depriv*liv_ind_weight) + (row.cooking_fuel_depriv*liv_ind_weight) + (row.floor_depriv*liv_ind_weight) + (row.asset_depriv*liv_ind_weight)

#eligible_df['total_of_weighted_deprivations']  = (eligible_df.school_attainment_depriv*edu_ind_weight) + (eligible_df.school_attendance_depriv*edu_ind_weight) + (eligible_df.malnutrition*health_ind_weight) + (eligible_df.child_mortailty*health_ind_weight) + (eligible_df.electricity_depriv*liv_ind_weight) + (eligible_df.water_depriv*liv_ind_weight) + (eligible_df.sanitation_depriv*liv_ind_weight) + (eligible_df.cooking_fuel_depriv*liv_ind_weight) + (eligible_df.floor_depriv*liv_ind_weight) + (eligible_df.asset_depriv*liv_ind_weight)
def calculate_deprivations(df, dhs_cluster_df, mp_threshold):
    # The headcount ratio, H, is the proportion of the multidimensionally poor in the population: H = q / n
    # where q is the number of people who are multidimensionally poor and n is the total population.
    df['headcount_poor'] =  np.where(df['total_of_weighted_deprivations'] >= mp_threshold, df['total_household_members'], 0)

    # The intensity of poverty, A, reflects the proportion of the weighted component indicators in which, on average,
    # poor people are deprived. For poor households only (deprivation score c of 33.3 percent or higher), the deprivation 
    # scores are summed and divided by the total number of poor people.
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

    # Calculate MPI at the required aggregate level
    df['headcount_ratio'] = df['headcount_poor']/df['total_household_members']
    df['poverty_intensity'] = df['total_poverty_intensity']/df['headcount_poor']
    df[col] = df['headcount_ratio'] * df['poverty_intensity']
    return df
# Function to combine MPI subnational scores with geometry
def get_mpi_subnational_gdf(mpi_subnational_df, states_provinces_gdf, country):
    # Join the mpi_subnational data to states and provinces data in order to plot

    # Keep just country data
    states_provinces_gdf = states_provinces_gdf[states_provinces_gdf['admin'] == country]
    mpi_subnational_df = mpi_subnational_df[mpi_subnational_df['Country'] == country]

    # This step is just to ensure we have matches where possible between the two datasets
    #from string import punctuation
    #states_provinces_gdf['name'] = states_provinces_gdf['name'].str.replace('-',' ')
    #mpi_subnational_df['Sub-national region'] = mpi_subnational_df['Sub-national region'].str.replace('-',' ')

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
# Function to run the whole process
# This calls all the subfunctions in order to calculate MPI at province level and calcualtes a correlation between 
# the country's calculated MPI and the OPHI MPI.
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

    eligible_df = get_combined_data_for_eligible_households()

    # calclate total weighted deprivations
    eligible_df['total_of_weighted_deprivations'] = eligible_df.apply(calculate_total_of_weighted_depriv, axis=1)

    # calculate MPI. mp_threshold is 0.333 because this is the cutoff for being considered multi-dimensionally poor 
    # (poor in more than one dimension, since there are 3 dimensions, this is 1/3)
    dhs_mpi_df = calculate_deprivations(eligible_df, dhs_cluster_df, 0.333)

    # Spatially join to admin1 boundaries
    #dhs_mpi_gdf = convert_to_geodataframe_with_lat_long(dhs_mpi_df, 'LONGNUM', 'LATNUM')
    #dhs_mpi_joined_gdf = gpd.sjoin(dhs_mpi_gdf, admin1_geo_gdf, op='within')
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
read_data('kenya', 
          '../input/kenya-preprocessed/kenya_household_dhs.csv',
          '../input/kenya-preprocessed/kenya_household_member_dhs.csv',
          '../input/kenya-preprocessed/kenya_births_dhs.csv',
          '../input/kenya-preprocessed/kenya_dhs_cluster.csv',
          '../input/kenya-preprocessed/KEGE71FL.shp', 
          '../input/kenya-humdata-admin-geo/Kenya_admin_2014_WGS84.shp', 
          '../input/kenya-humdata-admin-geo/KEN_Adm2.shp')
# Replace polygons with simple ones
# This step was done because folium maps were not plotting the original polygons for some reason. Maybe they were just too complex.

# This step is only necessary for certain shapefiles, when the geometry has too many points.
def replace_geometry(gdf, gdf_simple_path):
    gdf_simple = gpd.read_file(gdf_simple_path)
    gdf['geometry'] = gdf_simple['geometry']

# Note: Be careful when simplifying shapefiles, that there are still matches for all admin level entries and 
# no Polygon is simpified down to an empty polygon.
replace_geometry(admin1_geo_gdf, '../input/kenya-humdata-admin-geo/Kenya_admin_2014_WGS84_simple.shp')
replace_geometry(admin2_geo_gdf, '../input/kenya-humdata-admin-geo/KEN_Adm2_simple.shp')
admin1_mpi_df, admin2_mpi_df, admin3_mpi_df = calculate_mpi('Kenya', admin1_geo_gdf, 'Province', 'mpi_admin1', 
        admin2_geo=admin2_geo_gdf, admin2_col='ADM1NAME', admin2_mpi_col='mpi_admin2', 
        admin3_col='Adm2Name', admin3_mpi_col='mpi_admin3')
# Merge 
country_mpi_subnational_gdf = get_mpi_subnational_gdf(mpi_subnational_df, states_provinces_gdf, 'Kenya')
admin1_mpi_merged_df = admin1_mpi_df.merge(country_mpi_subnational_gdf[['Sub-national region', 'MPI Regional']],
                                                left_on=['Province'], right_on=['Sub-national region'])
print("Dataset after merge with OPHI MPI data: ", admin1_mpi_merged_df.shape)

# Check Correlation at admin1 level
print("Correlation, p-value: ", pearsonr(admin1_mpi_merged_df.loc[:, 'mpi_admin1'], admin1_mpi_merged_df.loc[:, 'MPI Regional']))
sns.regplot(x="MPI Regional", y='mpi_admin1', data=admin1_mpi_merged_df)
# RMSE
print("RMSE: ", np.sqrt(metrics.mean_squared_error(admin1_mpi_merged_df.loc[:, 'MPI Regional'], admin1_mpi_merged_df.loc[:, 'mpi_admin1'])))
# Define some helper functions
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
kenya_mpi_threshold_scale = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6] # Define MPI scale for Kenya
kenya_geo_gdf = get_geo_gdf('Kenya')
create_map(kenya_geo_gdf, country_mpi_subnational_gdf, 'feature.properties.name', 'Sub-national region', 'MPI Regional', 'YlOrRd', 0.0236, 37.9062, 6, kenya_mpi_threshold_scale)
create_map(kenya_geo_gdf, admin1_mpi_df, 'feature.properties.name', 'Province', 'mpi_admin1', 'YlOrRd', 0.0236, 37.9062, 6, kenya_mpi_threshold_scale)
create_map(admin1_geo_gdf, admin2_mpi_df, 'feature.properties.COUNTY', 'ADM1NAME', 'mpi_admin2', 'YlOrRd', 0.0236, 37.9062, 6, kenya_mpi_threshold_scale)
create_map(admin2_geo_gdf, admin3_mpi_df, 'feature.properties.Adm2Name', 'Adm2Name', 'mpi_admin3', 'YlOrRd', 0.0236, 37.9062, 6, kenya_mpi_threshold_scale)
admin2_mpi_df.min()
admin3_mpi_df.min()
# Original Kiva datasets
kiva_loans_df = pd.read_csv("../input/data-science-for-good-kiva-crowdfunding/kiva_loans.csv")
#kiva_mpi_locations_df = pd.read_csv("../input/data-science-for-good-kiva-crowdfunding/kiva_mpi_region_locations.csv")
loan_theme_ids_df = pd.read_csv("../input/data-science-for-good-kiva-crowdfunding/loan_theme_ids.csv")
loan_themes_by_region_df = pd.read_csv("../input/data-science-for-good-kiva-crowdfunding/loan_themes_by_region.csv")
# Merge Kiva loans to locations data via loan_themes
print("Original Kiva Loans dataset: ", kiva_loans_df.shape)
kiva_loans_region_df = pd.merge(kiva_loans_df, loan_theme_ids_df, how='left', on='id', suffixes=('', '_y'))
kiva_loans_region_df = kiva_loans_region_df[kiva_loans_region_df.columns.drop(list(kiva_loans_region_df.filter(regex='_y')))]

kiva_loans_region_df = kiva_loans_region_df.merge(loan_themes_by_region_df, how='left', on=['Partner ID', 'Loan Theme ID', 'country', 'region'], suffixes=('', '_y'))
kiva_loans_region_df = kiva_loans_region_df[kiva_loans_region_df.columns.drop(list(kiva_loans_region_df.filter(regex='_y')))]

#kiva_loans_region_df = kiva_loans_region_df.merge(kiva_loans_region_df, how='left', left_on=['country', 'mpi_region'], right_on=['country', 'LocationName'], suffixes=('', '_y'))
#kiva_loans_region_df = kiva_loans_region_df[kiva_loans_region_df.columns.drop(list(kiva_loans_region_df.filter(regex='_y')))]
print("Merged Kiva Loans dataset: ", kiva_loans_region_df.shape)
# Keep only Kenya loans
kiva_loans_kenya_df = kiva_loans_region_df[kiva_loans_region_df['country']=='Kenya']
print("Kenya Kiva Loans dataset: ", kiva_loans_kenya_df.shape)

# Drop those with null lat/long. If lat/long is not known, the borrower cannot be more accurately classified using this method.
kiva_loans_kenya_df = kiva_loans_kenya_df[np.isfinite(kiva_loans_kenya_df['lat'])]
print("Kenya cleaned Kiva Loans dataset: ", kiva_loans_kenya_df.shape)
# Get county geomentry

# Doing some manual recoding to get matches 
admin1_geo_gdf.COUNTY.replace('Keiyo-Marakwet', 'Elgeyo Marakwet', inplace=True)
admin1_geo_gdf.COUNTY.replace('Tharaka', 'Tharaka-Nithi', inplace=True)
admin1_geo_gdf.COUNTY.replace('Trans Nzoia', 'Trans-Nzoia', inplace=True)

print("Original dhs_mpi_county_df dataset: ", admin2_mpi_df.shape)
admin2_mpi_df = admin2_mpi_df.merge(admin1_geo_gdf[['COUNTY', 'Province', 'geometry']], left_on='ADM1NAME', right_on=['COUNTY'])
print("Merged dhs_mpi_county_df dataset: ", admin2_mpi_df.shape)
# function to add markers to folium map
def add_markers(df, m, radius='count', color='blue', popup=None):
    for i in range(0, df.shape[0]):
        folium.CircleMarker(
            [df.iloc[i]['lat'], df.iloc[i]['lon']], 
            radius=df.iloc[i][radius]**(4**-1), # x**(n**-1) is used because there are clusters of loans with the same lot/long
                                                 # and also single loans with unique lat/long
            color=color, 
            fill=True, 
            fill_color=color
        ).add_to(m)
# Convert to geo dataframe
kiva_loans_kenya_gdf = convert_to_geodataframe_with_lat_long(kiva_loans_kenya_df, 'lon', 'lat')

# Group by lat/long, count because there are many loans recorded with the same lat/long and it takes too long to plot otherwise.
print("Original kiva_loans_kenya_gdf dataset: ", kiva_loans_kenya_gdf.shape)
kiva_loans_kenya_grouped_gdf = kiva_loans_kenya_gdf.groupby(['lat','lon']).size().reset_index(name='count')
print("Grouped kiva_loans_kenya_gdf dataset: ", kiva_loans_kenya_grouped_gdf.shape)

# plot at county level
kenya_map = create_map(admin1_geo_gdf, admin2_mpi_df, 'feature.properties.COUNTY', 'ADM1NAME', 'mpi_admin2', 'YlOrRd', 0.0236, 37.9062, 6, kenya_mpi_threshold_scale)
add_markers(kiva_loans_kenya_grouped_gdf, kenya_map)
kenya_map
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
admin1_mpi_df, admin2_mpi_df, admin3_mpi_df = calculate_mpi('Zimbabwe', admin1_geo_gdf, 'ADM1NAME', 'mpi_admin1', admin2_geo=admin2_geo_gdf, admin2_col='DIST_NM_LA', admin2_mpi_col='mpi_admin2', )
# Merge 
country_mpi_subnational_gdf = get_mpi_subnational_gdf(mpi_subnational_df, states_provinces_gdf, 'Zimbabwe')
admin1_mpi_merged_df = admin1_mpi_df.merge(country_mpi_subnational_gdf[['Sub-national region', 'MPI Regional']],
                                                left_on=['ADM1NAME'], right_on=['Sub-national region'])
print("Dataset after merge with OPHI MPI data: ", admin1_mpi_merged_df.shape)

# Check Correlation at admin1 level
print("Correlation, p-value: ", pearsonr(admin1_mpi_merged_df.loc[:, 'mpi_admin1'], admin1_mpi_merged_df.loc[:, 'MPI Regional']))
sns.regplot(x="MPI Regional", y='mpi_admin1', data=admin1_mpi_merged_df)
# RMSE
print("RMSE: ", np.sqrt(metrics.mean_squared_error(admin1_mpi_merged_df.loc[:, 'MPI Regional'], admin1_mpi_merged_df.loc[:, 'mpi_admin1'])))
zmbabwe_mpi_threshold_scale = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3] # Define MPI scale for Zimbabwe
zimbabwe_geo_gdf = get_geo_gdf('Zimbabwe')
create_map(zimbabwe_geo_gdf, country_mpi_subnational_gdf, 'feature.properties.name', 'Sub-national region', 
           'MPI Regional', 'YlOrRd', -19.0154, 29.1549, 6, zmbabwe_mpi_threshold_scale)
create_map(zimbabwe_geo_gdf, admin1_mpi_df, 'feature.properties.name', 'ADM1NAME', 'mpi_admin1', 'YlOrRd', -19.0154, 29.1549, 6, zmbabwe_mpi_threshold_scale)
create_map(admin2_geo_gdf, admin2_mpi_df, 'feature.properties.DIST_NM_LA', 'DIST_NM_LA', 'mpi_admin2', 'YlOrRd', -19.0154, 29.1549, 6, zmbabwe_mpi_threshold_scale)
# Uncomment the line below to run pre-processing of original DHS files
#preprocess_dhs_data('cambodia', 'KHHR73FL.DTA', 'KHPR73FL.DTA', 'KHBR73FL.DTA', 'KHGC71FL.csv')
# Read in DHS and Geo Data
read_data('cambodia', 
          '../input/cambodia-preprocessed/cambodia_household_dhs.csv',
          '../input/cambodia-preprocessed/cambodia_household_member_dhs.csv',
          '../input/cambodia-preprocessed/cambodia_births_dhs.csv',
          '../input/cambodia-preprocessed/cambodia_dhs_cluster.csv',
          '../input/cambodia-preprocessed/KHGE71FL.shp', 
          '../input/cambodia-humdata-admin-geo/khm_admbnda_adm1_gov.shp', 
          '../input/cambodia-humdata-admin-geo/khm_admbnda_adm2_gov.shp'
         )

# Simplify geometry.
replace_geometry(admin1_geo_gdf, '../input/cambodia-humdata-admin-geo/khm_admbnda_adm1_gov_simple.shp')
replace_geometry(admin2_geo_gdf, '../input/cambodia-humdata-admin-geo/khm_admbnda_adm2_gov_simple.shp')
# Doing some manual recoding to get matches - no match for Banteay Mean Chey, Kampong Speu,  Kampong Thom, Kandal, Kratie, Pursat
states_provinces_gdf.name.replace('Bântéay Méanchey', 'Banteay Mean Chey', inplace=True)
#states_provinces_gdf.name.replace('Battambang & Pailin', 'Pailin', inplace=True)
states_provinces_gdf.name.replace('Batdâmbâng', 'Battambang & Pailin', inplace=True)
states_provinces_gdf.name.replace('Kâmpóng Cham', 'Kampong Cham', inplace=True)
states_provinces_gdf.name.replace('Kâmpóng Chhnang', 'Kampong Chhnang', inplace=True)
states_provinces_gdf.name.replace('Kâmpóng Spœ', 'Kampong Speu', inplace=True)
states_provinces_gdf.name.replace('Kâmpóng Thum', 'Kampong Thom', inplace=True)
states_provinces_gdf.name.replace('Kep', 'Kampot & Kep', inplace=True)
states_provinces_gdf.name.replace('Kândal', 'Kandal', inplace=True)
states_provinces_gdf.name.replace('Krâchéh', 'Kratie', inplace=True)
states_provinces_gdf.name.replace('Môndól Kiri', 'Mondol Kiri & Rattanak Kiri', inplace=True)
states_provinces_gdf.name.replace('Krong Preah Sihanouk', 'Preah Sihanouk & Kaoh Kong', inplace=True)
states_provinces_gdf.name.replace('Preah Vihéar', 'Preah Vihear & Steung Treng', inplace=True)
states_provinces_gdf.name.replace('Stœng Trêng', 'Preah Vihear & Steung Treng', inplace=True)
states_provinces_gdf.name.replace('Prey Vêng', 'Prey Veng', inplace=True)
states_provinces_gdf.name.replace('Pouthisat', 'Pursat', inplace=True)
states_provinces_gdf.name.replace('Siemréab', 'Siem Reap', inplace=True)
states_provinces_gdf.name.replace('Takêv', 'Takeo', inplace=True)

#cambodia_geo_gdf.name.replace('Bântéay Méanchey', 'Banteay Mean Chey', inplace=True)
#cambodia_geo_gdf.name.replace('Krong Pailin', 'Pailin', inplace=True)
#cambodia_geo_gdf.name.replace('Kaôh Kong', 'Preah Sihanouk & Kaoh Kon', inplace=True)
#cambodia_geo_gdf.name.replace('Stœng Trêng', 'Preah Vihear & Steung Treng', inplace=True)
admin1_mpi_df, admin2_mpi_df, admin3_mpi_df = calculate_mpi('Cambodia', admin1_geo_gdf, 'ADM1NAME', 'mpi_admin1', admin2_geo=admin2_geo_gdf, admin2_col='DIS_NAME', admin2_mpi_col='mpi_admin2')
# Merge 
country_mpi_subnational_gdf = get_mpi_subnational_gdf(mpi_subnational_df, states_provinces_gdf, 'Cambodia')
admin1_mpi_merged_df = admin1_mpi_df.merge(country_mpi_subnational_gdf[['Sub-national region', 'MPI Regional']],
                                                left_on=['ADM1NAME'], right_on=['Sub-national region'])
print("Dataset after merge with OPHI MPI data: ", admin1_mpi_merged_df.shape)

# Check Correlation at admin1 level
print("Correlation, p-value: ", pearsonr(admin1_mpi_merged_df.loc[:, 'mpi_admin1'], admin1_mpi_merged_df.loc[:, 'MPI Regional']))
sns.regplot(x="MPI Regional", y='mpi_admin1', data=admin1_mpi_merged_df)
# RMSE
print("RMSE: ", np.sqrt(metrics.mean_squared_error(admin1_mpi_merged_df.loc[:, 'MPI Regional'], admin1_mpi_merged_df.loc[:, 'mpi_admin1'])))
cambodia_mpi_threshold_scale = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3] # Define MPI scale
cambodia_geo_gdf = get_geo_gdf('Cambodia')
create_map(cambodia_geo_gdf, country_mpi_subnational_gdf, 'feature.properties.name', 'Sub-national region', 
           'MPI Regional', 'YlOrRd', 12.5657, 104.9910, 7, cambodia_mpi_threshold_scale)
create_map(cambodia_geo_gdf, admin1_mpi_df, 'feature.properties.name', 'ADM1NAME', 'mpi_admin1', 'YlOrRd',  12.5657, 104.9910, 7, cambodia_mpi_threshold_scale)
create_map(admin2_geo_gdf, admin2_mpi_df, 'feature.properties.DIS_NAME', 'DIS_NAME', 'mpi_admin2', 'YlOrRd',  12.5657, 104.9910, 7, cambodia_mpi_threshold_scale)
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
admin1_mpi_df, admin2_mpi_df, admin3_mpi_df = calculate_mpi('Rwanda', admin1_geo_gdf, 'ADM1NAME', 'mpi_admin1', admin2_geo=admin2_geo_gdf, admin2_col='NOMDISTR', admin2_mpi_col='mpi_admin2')
# Merge 
country_mpi_subnational_gdf = get_mpi_subnational_gdf(mpi_subnational_df, states_provinces_gdf, 'Rwanda')
admin1_mpi_merged_df = admin1_mpi_df.merge(country_mpi_subnational_gdf[['Sub-national region', 'MPI Regional']],
                                                left_on=['ADM1NAME'], right_on=['Sub-national region'])
print("Dataset after merge with OPHI MPI data: ", admin1_mpi_merged_df.shape)

# Check Correlation at admin1 level
print("Correlation, p-value: ", pearsonr(admin1_mpi_merged_df.loc[:, 'mpi_admin1'], admin1_mpi_merged_df.loc[:, 'MPI Regional']))
sns.regplot(x="MPI Regional", y='mpi_admin1', data=admin1_mpi_merged_df)
# RMSE
print("RMSE: ", np.sqrt(metrics.mean_squared_error(admin1_mpi_merged_df.loc[:, 'MPI Regional'], admin1_mpi_merged_df.loc[:, 'mpi_admin1'])))
rwanda_mpi_threshold_scale = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3] # Define MPI scale
rwanda_geo_gdf = get_geo_gdf('Rwanda')
create_map(rwanda_geo_gdf, country_mpi_subnational_gdf, 'feature.properties.name', 'Sub-national region', 
           'MPI Regional', 'YlOrRd', -1.9403, 29.8739, 8, rwanda_mpi_threshold_scale)
create_map(rwanda_geo_gdf, admin1_mpi_df, 'feature.properties.name', 'ADM1NAME', 'mpi_admin1', 'YlOrRd', -1.9403, 29.8739, 8, rwanda_mpi_threshold_scale)
# Ensure we are using title case for names (this is inconsistent in some country's datasets)
admin2_geo_gdf['NOMDISTR'] = admin2_geo_gdf['NOMDISTR'].str.title()
create_map(admin2_geo_gdf, admin2_mpi_df, 'feature.properties.NOMDISTR', 'NOMDISTR', 'mpi_admin2', 'YlOrRd',  -1.9403, 29.8739, 8, rwanda_mpi_threshold_scale)
# Uncomment the line below to run pre-processing of original DHS files
#preprocess_dhs_data('tanzania', 'TZHR7HFL.DTA', 'TZPR7HFL.DTA', 'TZBR7HFL.DTA', 'TZGC7AFL.csv')
# Read in DHS and Geo Data
read_data('tanzania', 
          '../input/tanzania-preprocessed/tanzania_household_dhs.csv',
          '../input/tanzania-preprocessed/tanzania_household_member_dhs.csv',
          '../input/tanzania-preprocessed/tanzania_births_dhs.csv',
          '../input/tanzania-preprocessed/tanzania_dhs_cluster.csv',
          '../input/tanzania-preprocessed/TZGE7AFL.shp', 
          '../input/tanzania-humdata-admin-geo/tza_popa_adm1_regions_TNBS2012_OCHA.shp', 
          '../input/tanzania-humdata-admin-geo/tza_popa_adm2_districts_TNBS2012_OCHA.shp')

# Simplify geometry. 
replace_geometry(admin2_geo_gdf, '../input/tanzania-humdata-admin-geo/tza_popa_adm2_districts_TNBS2012_OCHA_simple.shp')
admin1_mpi_df, admin2_mpi_df, admin3_mpi_df = calculate_mpi('Tanzania', admin1_geo_gdf, 'REGION', 'mpi_admin1', admin2_geo=admin2_geo_gdf, admin2_col='DISTRICT', admin2_mpi_col='mpi_admin2')
tanzania_mpi_threshold_scale = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]# Define MPI scale
tanzania_geo_gdf = get_geo_gdf('Tanzania')
create_map(tanzania_geo_gdf, admin1_mpi_df, 'feature.properties.name', 'REGION', 'mpi_admin1', 'YlOrRd', -6.3690, 34.8888, 6, tanzania_mpi_threshold_scale)
create_map(admin2_geo_gdf, admin2_mpi_df, 'feature.properties.DISTRICT', 'DISTRICT', 'mpi_admin2', 'YlOrRd', -6.3690, 34.8888, 6, tanzania_mpi_threshold_scale)
# Uncomment the line below to run pre-processing of original DHS files
#preprocess_dhs_data('ghana', 'GHHR72FL.DTA', 'GHPR72FL.DTA', 'GHBR72FL.DTA', 'GHGC71FL.csv')
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
admin1_mpi_df, admin2_mpi_df, admin3_mpi_df = calculate_mpi('Ghana', admin1_geo_gdf, 'ADM1_NAME', 'mpi_admin1', admin2_geo=admin2_geo_gdf, admin2_col='ADM2_NAME', admin2_mpi_col='mpi_admin2')
# Merge 
country_mpi_subnational_gdf = get_mpi_subnational_gdf(mpi_subnational_df, states_provinces_gdf, 'Ghana')
admin1_mpi_merged_df = admin1_mpi_df.merge(country_mpi_subnational_gdf[['Sub-national region', 'MPI Regional']],
                                                left_on=['ADM1_NAME'], right_on=['Sub-national region'])
print("Dataset after merge with OPHI MPI data: ", admin1_mpi_merged_df.shape)

# Check Correlation at admin1 level
print("Correlation, p-value: ", pearsonr(admin1_mpi_merged_df.loc[:, 'mpi_admin1'], admin1_mpi_merged_df.loc[:, 'MPI Regional']))
sns.regplot(x="MPI Regional", y='mpi_admin1', data=admin1_mpi_merged_df)
# RMSE
print("RMSE: ", np.sqrt(metrics.mean_squared_error(admin1_mpi_merged_df.loc[:, 'MPI Regional'], admin1_mpi_merged_df.loc[:, 'mpi_admin1'])))
ghana_mpi_threshold_scale = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
ghana_geo_gdf = get_geo_gdf('Ghana')
create_map(ghana_geo_gdf, country_mpi_subnational_gdf, 'feature.properties.name', 'Sub-national region', 
           'MPI Regional', 'YlOrRd', 7.9465, -1.0232, 6, ghana_mpi_threshold_scale)
create_map(ghana_geo_gdf, admin1_mpi_df, 'feature.properties.name', 'ADM1_NAME', 'mpi_admin1', 'YlOrRd', 7.9465, -1.0232, 6, ghana_mpi_threshold_scale)
create_map(admin2_geo_gdf, admin2_mpi_df, 'feature.properties.ADM2_NAME', 'ADM2_NAME', 'mpi_admin2', 'YlOrRd', 7.9465, -1.0232, 6, ghana_mpi_threshold_scale)