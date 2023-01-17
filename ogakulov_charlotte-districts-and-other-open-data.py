import pandas as pd
import seaborn as sns

# Looking at Boston police interrogation and observation files
individuals = pd.read_csv('../input/cpe-external-data/cpe_external_data/cpe_external_data/1-Boston/Boston_Field_Contact_Demographics_2016.csv')
calls = pd.read_csv('../input/cpe-external-data/cpe_external_data/cpe_external_data/1-Boston/Boston_Field_Contact_Report_2016.csv')
print('There were {} Boston police interactions with individuals in 2016'.format(individuals.shape[0]))
# Quick bargraph of the race distribution
sns.countplot(y='race', data=individuals, orient='h');
# Looking at Boston police interrogation and observation files
ind_uof = pd.read_csv('../input/cpe-external-data/cpe_external_data/cpe_external_data/2-Indianapolis/Indianapolis_Police_Use_of_Force.csv')
ind_psi = pd.read_csv('../input/cpe-external-data/cpe_external_data/cpe_external_data/2-Indianapolis/Indianapolis_Police_Shootings.csv')
print('There are {} records in Use of Force file for Indianapolis, from 2014 to 2018'.format(ind_uof.shape[0]))
# Convert date column from string to datetime
ind_uof['occurredDate'] = pd.to_datetime(ind_uof['occurredDate'])

# Plot distribution by race of incidents in 2016
sns.countplot(y='residentRace', data=ind_uof[ind_uof['occurredDate'].dt.year == 2016], orient='h');
# Read and plot the shapefile
import geopandas as gpd 

charlotte_map = gpd.GeoDataFrame.from_file('../input/cpe-external-data/cpe_external_data/cpe_external_data/3-Charlotte/CMPD_Police_Divisions.shp')
charlotte_map.plot();
# Load Austin datasets
aus_ois = pd.read_csv('../input/cpe-external-data/cpe_external_data/cpe_external_data/4-Austin/Austin_2008-17_OIS_Incidents.csv')
aus_ois_officers = pd.read_csv('../input/cpe-external-data/cpe_external_data/cpe_external_data/4-Austin/Austin_2008-17_OIS_Officers.csv')
aus_ois_subjects = pd.read_csv('../input/cpe-external-data/cpe_external_data/cpe_external_data/4-Austin/Austin_2008-17_OIS_Subjects.csv')
sns.countplot(y='Subject Race/Ethnicity', data=aus_ois_subjects, orient='h');
# Load Dallas datasets
dal_ois = pd.read_csv('../input/cpe-external-data/cpe_external_data/cpe_external_data/5-Dallas/Dallas_Police_Officer-Involved_Shootings.csv')
dal_uof = pd.read_csv('../input/cpe-external-data/cpe_external_data/cpe_external_data/5-Dallas/Police_Response_to_Resistance_-_2016.csv')
sns.countplot(y='CitRace', data=dal_uof, orient='h');
dal_ois.head()
# Load Seattle datasets
sea_ois = pd.read_csv('../input/cpe-external-data/cpe_external_data/cpe_external_data/6-Seattle/Seattle_Officer_Involved_Shooting.csv')
sea_uof = pd.read_csv('../input/cpe-external-data/cpe_external_data/cpe_external_data/6-Seattle/Seattle_Use_Of_Force.csv')
sns.countplot(y='Subject_Race', data=sea_uof, orient='h');