# importing usual libs
import plotly.offline as py
import seaborn as sns
import pandas as pd
import numpy as np
import zipfile
import os

# importing only specific objects from these libs to save up memory
from datetime import timedelta as dt_timedelta
from datetime import date as dt_date
from time import time as t_time
from gc import collect as gc_collect

from string import punctuation as str_punctuation
from nltk import edit_distance

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import *
from sklearn.cluster import KMeans
from sklearn.ensemble import *
from sklearn.tree import *

from matplotlib import patheffects as path_effects
from matplotlib import pyplot as plt
from mpl_toolkits.basemap import maskoceans
from mpl_toolkits.basemap import Basemap

plt.style.use('fivethirtyeight')
py.init_notebook_mode()
raw_start = t_time()
start = t_time()
kiva_loans = pd.read_csv('../input/data-science-for-good-kiva-crowdfunding/kiva_loans.csv',
                         parse_dates=['posted_time', 'disbursed_time', 'funded_time']
                        ).drop(columns=['country_code', 'date', 'tags', 'use'])
print('Number of loans: {}'.format(kiva_loans.shape[0]))
print('Time taken: {:.3f}s'.format(t_time()-start))
def timer(given_start=None):
    if given_start is None:
        print('Time taken: {:.3f}s'.format(t_time()-start))
    else:
        print('Time taken: {:.3f}s'.format(t_time()-given_start))

def raw_timer():
    print('\nTotal time taken to run the kernel: {:.3f}m'.format((t_time()-raw_start)/60))

def rows_with_countries(df):
    return df[df['country'].notna()]

def rows_with_countries_and_regions(df):
    return df[(df['country'].notna()) & (df['region'].notna())]

def rows_with_coordinates(df):
    return df[(df['longitude'].notna()) & (df['latitude'].notna())]

def rows_without_coordinates(df):
    return df[(df['longitude'].isna()) | (df['latitude'].isna())]

def rows_with_loans(df):
    return df[df['posted_time'].notna()]

def report(columns=None):
    timer()
    n_loans = rows_with_loans(kiva_loans).shape[0]
    if n_loans==671205:
        print('Loans are intact (671205)', end='\n' if columns is None else '\n\n')
    elif n_loans>671205:
        print('{} loans have been duplicated\n'.format(n_loans-671205), end='\n' if columns is None else '\n\n')
    else:
        print('{} loans are missing.\n'.format(671205-n_loans))
    if not columns is None:
        kiva_loans[['country']+columns].info()

kiva_country_names = {
    'Democratic Republic of the Congo': 'The Democratic Republic of the Congo',
    'Congo, Democratic Republic of the': 'The Democratic Republic of the Congo',
    'Republic of the Congo': 'Congo',
    'Congo, Republic of': 'Congo',
    'Myanmar': 'Myanmar (Burma)',
    'Ivory Coast': "Cote D'Ivoire",
    "Cote d'Ivoire": "Cote D'Ivoire",
    'VietNam': 'Vietnam',
    'Laos': "Lao People's Democratic Republic",
    'Bolivia, Plurinational State of': 'Bolivia',
    'Palestinian Territories': 'Palestine',
    'Somaliland region': 'Somalia',
    'Syrian Arab Republic': 'Syria',
    'Tanzania, United Republic of': 'Tanzania',
    'Viet Nam': 'Vietnam',
    'Palestine, State ofa': 'Palestine',
    'Macedonia, The former Yugoslav Republic of': 'Macedonia',
    'Moldova, Republic of': 'Moldova'
}

def get_standardized_country_name(obj):
    if type(obj) is float:
        return np.nan
    if obj in kiva_country_names:
        return kiva_country_names[obj]
    return obj

def get_standardized_region_name(obj):
    if type(obj) is float:
        return np.nan
    chars = [char if char not in str_punctuation else ' ' for char in obj.lower()]
    obj = ''.join(chars)
    words = obj.split(' ')
    removed_words = ['', 'province', 'village', 'ville', 'district', 'town', 'city', 'region', 'reagion',
                     'community', 'commune', 'comunidad', 'odisha', 'aldea', 'kampong', 'kompong', 'ciudad',
                    'jalalabad', 'jalalabat', 'localidad']
    words = [word for word in words if word not in removed_words]
    return ' '.join(words).strip().title()

def standardize_dataset(df):
    df['country'] = df['country'].apply(get_standardized_country_name)
    if 'region' in df.columns:
        df['region'] = df['region'].apply(get_standardized_region_name)
start = t_time()
standardize_dataset(kiva_loans)
timer()
start = t_time()
kiva_loans['borrower_count'] = kiva_loans['borrower_genders'].apply(lambda x: np.nan if type(x) is float else len(x.split(',')))
kiva_loans['borrower_female_pct'] = kiva_loans['borrower_genders'].apply(lambda x: np.nan if type(x) is float else x.count('female'))/kiva_loans['borrower_count']

kiva_loans.drop(columns=['borrower_genders'], inplace=True)
gc_collect()
report(['borrower_count', 'borrower_female_pct'])
start = t_time()
kiva_loans['posting_delay_in_days'] = ((kiva_loans['posted_time'] - kiva_loans['disbursed_time'])/np.timedelta64(1, 's'))/86400
kiva_loans['posting_delay_in_days'] = kiva_loans['posting_delay_in_days'].apply(lambda x: max(0, x))
kiva_loans['funding_delay_in_days'] = ((kiva_loans['funded_time'] - kiva_loans['posted_time'])/np.timedelta64(1, 's'))/86400
kiva_loans['total_delay_in_days'] = kiva_loans['posting_delay_in_days'] + kiva_loans['funding_delay_in_days']
report(['posting_delay_in_days', 'funding_delay_in_days', 'total_delay_in_days'])
start = t_time()
kiva_loans['funded_amount_per_lender'] = kiva_loans['funded_amount']/kiva_loans['lender_count']
kiva_loans['funded_amount_per_day'] = kiva_loans['funded_amount']/kiva_loans['funding_delay_in_days']
kiva_loans['loan_amount_per_borrower'] = kiva_loans['loan_amount']/kiva_loans['borrower_count']
kiva_loans['loan_amount_per_month'] = kiva_loans['loan_amount']/kiva_loans['term_in_months']
report(['funded_amount_per_lender', 'funded_amount_per_day', 'loan_amount_per_borrower', 'loan_amount_per_month'])
start = t_time()
kiva_loans['missing_funds'] = kiva_loans['loan_amount'] - kiva_loans['funded_amount']
kiva_loans['missing_funds_pct'] = kiva_loans['missing_funds']/kiva_loans['loan_amount']
report(['missing_funds', 'missing_funds_pct'])
start = t_time()
country_stats = pd.read_csv('../input/additional-kiva-snapshot/country_stats.csv'
                           ).drop(columns=['country_name', 'country_code', 'country_code3']
                                 ).rename(columns={'kiva_country_name':'country', 'region':'continent_region'})
standardize_dataset(country_stats)
kiva_loans = pd.merge(left=kiva_loans, right=rows_with_countries(country_stats), how='outer', on='country')
del country_stats
gc_collect()
report(['continent', 'continent_region', 'population', 'population_below_poverty_line', 'hdi', 'life_expectancy', 'expected_years_of_schooling', 'mean_years_of_schooling', 'gni'])
start = t_time()
kiva_loans['gni_per_capta'] = kiva_loans['gni']/kiva_loans['population']
kiva_loans.drop(columns=['population', 'gni'], inplace=True)
kiva_loans['mean_years_of_schooling_pct'] = kiva_loans['mean_years_of_schooling']/kiva_loans['expected_years_of_schooling']
gc_collect()
report(['gni_per_capta', 'mean_years_of_schooling_pct'])
delta = (kiva_loans['funded_time'].max() - kiva_loans['disbursed_time'].min()).total_seconds()/2
print('Date occurrences center: ~{}'.format(kiva_loans['disbursed_time'].min() + dt_timedelta(seconds=delta)))
start = t_time()
happiness_scores = pd.read_csv('../input/world-happiness/2015.csv',
                               usecols=['Country', 'Happiness Score']
                              ).rename(columns={'Country':'country', 'Happiness Score':'happiness'})
standardize_dataset(happiness_scores)
kiva_loans = pd.merge(left=kiva_loans,
                      right=rows_with_countries(happiness_scores)[(happiness_scores['country']!='Congo (Brazzaville)') & (happiness_scores['country']!='Congo (Kinshasa)')],
                      how='outer', on='country')

kiva_loans.loc[(kiva_loans['country']=='Congo') & (kiva_loans['region']=='brazzaville'), 'happiness'] = happiness_scores.loc[happiness_scores['country']=='Congo (Brazzaville)', 'happiness'].values[0]
kiva_loans.loc[(kiva_loans['country']=='Congo') & (kiva_loans['region']=='kinshasa'), 'happiness'] = happiness_scores.loc[happiness_scores['country']=='Congo (Kinshasa)', 'happiness'].values[0]

del happiness_scores
gc_collect()
report(['happiness'])
start = t_time()
gpi = pd.read_csv('../input/gpi2008-2016/gpi_2008-2016.csv')
gpi['score_2013'].fillna(gpi['score_2016'], inplace=True)
gpi['score_2014'].fillna(gpi['score_2016'], inplace=True)
gpi['score_2015'].fillna(gpi['score_2016'], inplace=True)

standardize_dataset(gpi)
kiva_loans = pd.merge(left=kiva_loans, right=rows_with_countries(gpi), how='outer', on='country')

kiva_loans['gpi'] = np.nan
for i in range(kiva_loans.shape[0]):
    year = min(2016,kiva_loans.at[i, 'posted_time'].year)
    kiva_loans.at[i, 'gpi'] = kiva_loans.at[i, 'score_'+str(year)]

kiva_loans.drop(columns=[column for column in kiva_loans if column.count('score_')>0], inplace=True)
del gpi
gc_collect()
report(['gpi'])
start = t_time()
loan_theme_ids = pd.read_csv('../input/data-science-for-good-kiva-crowdfunding/loan_theme_ids.csv').drop(columns=['Loan Theme ID', 'Partner ID'])
kiva_loans = pd.merge(left=kiva_loans, right=loan_theme_ids, how='left', on='id')
del loan_theme_ids
gc_collect()
report(['Loan Theme Type'])
start = t_time()
loan_coords = pd.read_csv('../input/additional-kiva-snapshot/loan_coords.csv').rename(columns={'loan_id':'id'})
kiva_loans = pd.merge(left=kiva_loans, right=loan_coords, how='left', on='id')
del loan_coords
gc_collect()
report(['longitude', 'latitude'])
def data_quality_report():
    start = t_time()
    n_loans = rows_with_loans(kiva_loans).shape[0]
    rows_with_countries_and_regions_df = rows_with_countries_and_regions(kiva_loans)
    n_regions = rows_with_countries_and_regions_df.shape[0]
    n_coordinates = rows_with_coordinates(kiva_loans).shape[0]
    n_unique_regions = len(set(rows_with_countries_and_regions_df['region']))
    
    print('Percentage of regions properly labeled: {:.3f}%'.format(100*n_regions/n_loans))
    print('Percentage of coordinates properly labeled: {:.3f}%'.format(100*n_coordinates/n_loans))
    print('Number of unique region names: {}\n'.format(n_unique_regions))
    timer(start)

data_quality_report()
start = t_time()
loan_themes_by_region = pd.read_csv('../input/data-science-for-good-kiva-crowdfunding/loan_themes_by_region.csv').rename(columns={'Partner ID':'partner_id'})
standardize_dataset(loan_themes_by_region)
timer()
start = t_time()
kiva_loans = pd.merge(left=kiva_loans,
                      right=loan_themes_by_region[['partner_id', 'Field Partner Name', 'rural_pct']].drop_duplicates(subset=['partner_id']),
                      how='left', on='partner_id').drop(columns=['partner_id'])
report(['Field Partner Name', 'rural_pct', 'longitude', 'latitude'])
start = t_time()
kiva_loans = pd.merge(left=kiva_loans,
                      right=rows_with_countries_and_regions(loan_themes_by_region)[['country', 'region', 'lon', 'lat']].drop_duplicates(subset=['country', 'region']),
                      how='left', on=['country', 'region'])
kiva_loans['longitude'].fillna(kiva_loans['lon'], inplace=True)
kiva_loans['latitude'].fillna(kiva_loans['lat'], inplace=True)
kiva_loans.drop(columns=['lon', 'lat'], inplace=True)
gc_collect()
report(['longitude', 'latitude', 'region'])
start = t_time()
loan_themes_by_region.rename(columns={'lon':'longitude', 'lat':'latitude', 'region':'region_new'}, inplace=True)
kiva_loans = pd.merge(left=kiva_loans,
                      right=rows_with_coordinates(loan_themes_by_region)[['longitude', 'latitude', 'region_new']].drop_duplicates(subset=['longitude', 'latitude']),
                      how='left', on=['longitude', 'latitude'])
kiva_loans['region'].fillna(kiva_loans['region_new'], inplace=True)
kiva_loans.drop(columns=['region_new'], inplace=True)
gc_collect()
report(['region'])
start = t_time()
loan_themes_by_region['forkiva'].replace(['Yes', 'No'], [1, 0], inplace=True)
kiva_loans = pd.merge(left=kiva_loans,
                      right=loan_themes_by_region[['Loan Theme Type', 'forkiva']].drop_duplicates(subset=['Loan Theme Type']),
                      how='left', on='Loan Theme Type')
del loan_themes_by_region
gc_collect()
report(['forkiva'])
start = t_time()
kiva_mpi_region_locations = pd.read_csv('../input/data-science-for-good-kiva-crowdfunding/kiva_mpi_region_locations.csv',
                                        usecols=['country', 'region', 'MPI', 'lon', 'lat']).rename(columns={'MPI':'MPI_regional'})
standardize_dataset(kiva_mpi_region_locations)
report(['longitude', 'latitude'])
start = t_time()
kiva_loans = pd.merge(left=kiva_loans,
                      right=rows_with_countries_and_regions(kiva_mpi_region_locations),
                      how='left', on=['country', 'region'])
kiva_loans['longitude'].fillna(kiva_loans['lon'], inplace=True)
kiva_loans['latitude'].fillna(kiva_loans['lat'], inplace=True)
kiva_loans.drop(columns=['lon', 'lat'], inplace=True)
report(['MPI_regional', 'longitude', 'latitude'])
start = t_time()
kiva_mpi_region_locations.rename(columns={'lon':'longitude', 'lat':'latitude', 'MPI_regional':'MPI_regional_new'}, inplace=True)
kiva_loans = pd.merge(left=kiva_loans,
                      right=rows_with_coordinates(kiva_mpi_region_locations).drop(columns=['country', 'region']).drop_duplicates(subset=['longitude', 'latitude']),
                      how='left', on=['longitude', 'latitude'])
kiva_loans['MPI_regional'].fillna(kiva_loans['MPI_regional_new'], inplace=True)
kiva_loans.drop(columns=['MPI_regional_new'], inplace=True)
del kiva_mpi_region_locations
gc_collect()
report(['MPI_regional'])
start = t_time()
MPI_subnational = pd.read_csv('../input/mpi/MPI_subnational.csv',
                              usecols=['Country',
                                       'MPI National']).drop_duplicates(subset=['Country']).rename(columns={'Country':'country',
                                                                                                            'MPI National':'MPI_national'})
standardize_dataset(MPI_subnational)
kiva_loans = pd.merge(left=kiva_loans, right=rows_with_countries(MPI_subnational), how='left', on='country')
del MPI_subnational
gc_collect()
report(['MPI_national'])
start = t_time()
loans_lenders = pd.read_csv('../input/additional-kiva-snapshot/loans_lenders.csv').rename(columns={'loan_id':'id'})
loans_lenders['lenders'] = loans_lenders['lenders'].apply(lambda x: x.replace(' ', ''))
kiva_loans = pd.merge(left=kiva_loans, right=loans_lenders, how='left', on='id').drop(columns=['id'])
del loans_lenders
gc_collect()
report(['lenders'])
start = t_time()
lenders = pd.read_csv('../input/additional-kiva-snapshot/lenders.csv', usecols=['permanent_name', 'member_since'])
lenders_dict = {}
for i in range(lenders.shape[0]):
    lenders_dict[lenders.at[i, 'permanent_name']] = lenders.at[i, 'member_since']
del lenders
gc_collect()
timer()
start = t_time()

def get_lender_member_since(permanent_name):
    if permanent_name in lenders_dict:
        return lenders_dict[permanent_name]
    return np.nan

def avg_member_since(obj):
    if type(obj) is float:
        return np.nan
    member_sinces = []
    permanent_names = obj.split(',')
    for permanent_name in permanent_names:
        member_sinces.append(get_lender_member_since(permanent_name))
    return np.array(member_sinces).mean()

kiva_loans['lenders_members_since_avg'] = kiva_loans['lenders'].apply(avg_member_since)
kiva_loans['lenders_experience_in_days_avg'] = ((kiva_loans['posted_time'] - dt_date(1970,1,1)) / np.timedelta64(1, 's') - kiva_loans['lenders_members_since_avg'])/86400
kiva_loans.drop(columns=['lenders_members_since_avg', 'lenders'], inplace=True)

del lenders_dict
gc_collect()

report(['lenders_experience_in_days_avg'])
start = t_time()

def normalize_feature(df, feature, decr=False):
    if decr:
        df[feature] = -1.0 * df[feature]
    df[feature] = df[feature]-df[feature].min()
    df[feature] = df[feature]/(df[feature].max()-df[feature].min())

indicators = ['population_below_poverty_line', 'hdi', 'life_expectancy', 'expected_years_of_schooling',
              'mean_years_of_schooling', 'gni_per_capta', 'mean_years_of_schooling_pct', 'happiness', 'gpi']

for indicator in indicators:
    if indicator in ['population_below_poverty_line', 'gpi']:
        normalize_feature(kiva_loans, indicator, decr=True)
    else:
        normalize_feature(kiva_loans, indicator)

timer()
countries_without_continent_regions = ', '.join([country for country in set(kiva_loans[kiva_loans['continent_region'].isna()]['country'])])
countries_without_continents = ', '.join([country for country in set(kiva_loans[kiva_loans['continent'].isna()]['country'])])

print('Countries without continent_region values: {}\n'.format(countries_without_continent_regions))
print('Countries without continent values: {}'.format(countries_without_continents))
start = t_time()

# Taiwan and North Cyprus
kiva_loans.loc[(kiva_loans['country']=='Taiwan') | (kiva_loans['country']=='North Cyprus'), 'continent'] = 'Asia'
kiva_loans.loc[kiva_loans['country']=='Taiwan', 'continent_region'] = 'Eastern Asia'
kiva_loans.loc[kiva_loans['country']=='North Cyprus', 'continent_region'] = 'Western Asia'

# Saint Vincent and the Grenadines and Virgin Islands
kiva_loans.loc[(kiva_loans['country']=='Saint Vincent and the Grenadines') | (kiva_loans['country']=='Virgin Islands'), 'continent'] = 'Americas'
kiva_loans.loc[(kiva_loans['country']=='Saint Vincent and the Grenadines') | (kiva_loans['country']=='Virgin Islands'), 'continent_region'] = 'Caribbean'

# Guam and Vanuatu
kiva_loans.loc[(kiva_loans['country']=='Guam') | (kiva_loans['country']=='Vanuatu'), 'continent'] = 'Oceania'
kiva_loans.loc[kiva_loans['country']=='Guam', 'continent_region'] = 'Micronesia'
kiva_loans.loc[kiva_loans['country']=='Vanuatu', 'continent_region'] = 'Melanesia'

report(['continent', 'continent_region', 'longitude', 'latitude'])
SIMILARITY_THRESHHOLD = 0.9

start = t_time()

all_regions = rows_with_countries_and_regions(kiva_loans[['country', 'region', 'longitude', 'latitude']]).groupby(['country', 'region']).head().drop_duplicates(subset=['country', 'region']).reset_index(drop=True)

regions_with_coordinates = rows_with_coordinates(all_regions).rename(columns={'region':'region_with_coordinates'})
regions_with_coordinates['region_with_coordinates[0]'] = regions_with_coordinates['region_with_coordinates'].apply(lambda x: x[0])
regions_without_coordinates = rows_without_coordinates(all_regions).rename(columns={'region':'region_without_coordinates'}).drop(columns=['longitude', 'latitude'])
regions_without_coordinates['region_without_coordinates[0]'] = regions_without_coordinates['region_without_coordinates'].apply(lambda x: x[0])

cartesian = pd.merge(left=regions_without_coordinates, right=regions_with_coordinates, how='inner', on='country')
cartesian = cartesian[cartesian['region_without_coordinates[0]']==cartesian['region_with_coordinates[0]']].drop(columns=['region_without_coordinates[0]', 'region_with_coordinates[0]']).reset_index(drop=True)
cartesian['region_names_similarity'] = np.nan
for i in range(cartesian.shape[0]):
    region1 = cartesian.at[i, 'region_with_coordinates']
    region2 = cartesian.at[i, 'region_without_coordinates']
    cartesian.at[i, 'region_names_similarity'] = 1 - edit_distance(region1, region2)/(len(region1)+len(region2))
cartesian.sort_values(by='region_names_similarity', ascending=False, inplace=True)
cartesian = cartesian[cartesian['region_names_similarity']>SIMILARITY_THRESHHOLD].drop_duplicates(subset=['region_without_coordinates'])

timer()
cartesian.head(10)
cartesian.tail(10)
start = t_time()

cartesian = cartesian.drop(columns=['region_with_coordinates', 'region_names_similarity']).rename(columns={'region_without_coordinates':'region', 'longitude':'longitude_new', 'latitude':'latitude_new'})
kiva_loans = pd.merge(left=kiva_loans, right=cartesian, how='left', on=['country', 'region'])
kiva_loans['longitude'].fillna(kiva_loans['longitude_new'], inplace=True)
kiva_loans['latitude'].fillna(kiva_loans['latitude_new'], inplace=True)
kiva_loans.drop(columns=['longitude_new', 'latitude_new'], inplace=True)

del cartesian, all_regions, regions_with_coordinates, regions_without_coordinates
gc_collect()

report(['longitude', 'latitude', 'region'])
start = t_time()

class Counter:
    def __init__(self):
        self.counter = 0
    def tick(self):
        self.counter += 1
        return str(self.counter-1)

unknown_id_counter = Counter()

def join_names(obj):
    unknown_name = 'Unknown-'+unknown_id_counter.tick()
    if type(obj) is float:
        return unknown_name
    obj = sorted([element if not type(element) is float else unknown_name for element in set(obj)])
    has_known_name = False
    for name in obj:
        if name.count('Unknown-')==0:
            has_known_name = True
            break
    if has_known_name:
        obj = [name for name in obj if name.count('Unknown-')==0]
    else:
        obj = [obj[0]]
    return '/'.join(obj)

new_region_names = rows_with_coordinates(kiva_loans).groupby(['country', 'longitude', 'latitude'])['region'].apply(join_names).reset_index()
kiva_loans.rename(columns={'region':'region_new'}, inplace=True)
kiva_loans = pd.merge(left=kiva_loans, right=new_region_names, how='left', on=['country', 'longitude', 'latitude'])
kiva_loans['region'].fillna(kiva_loans['region_new'], inplace=True)
kiva_loans.drop(columns=['region_new'], inplace=True)

del new_region_names
gc_collect()

report(['region'])
start = t_time()

def rename_country(old_country, region, new_country):
    kiva_loans.loc[(kiva_loans['country']==old_country) & (kiva_loans['region']==region), 'country'] = new_country

def replace_numeric_features_with_groupby_mean(groupby, fixing_coordinates=False):
    group = kiva_loans.dropna(subset=groupby)[groupby + numeric_features].groupby(groupby).mean().reset_index().rename(columns=numeric_features_map)
    if not fixing_coordinates:
        group['borrower_female_pct_new'] = group['borrower_female_count_new']/group['borrower_count_new']
        group['missing_funds_pct_new'] = group['missing_funds_new']/group['loan_amount_new']
    merge = pd.merge(left=kiva_loans, right=group, how='left', on=groupby)
    for feature in numeric_features:
        kiva_loans[feature].fillna(merge[numeric_features_map[feature]], inplace=True)
    del merge, group
    gc_collect()

kiva_loans['borrower_female_count'] = kiva_loans['borrower_female_pct']*kiva_loans['borrower_count']
numeric_features = [column for column in kiva_loans.drop(columns=['longitude', 'latitude', 'MPI_regional',
                    'MPI_national']).columns if kiva_loans[column].dtype==np.int64 or kiva_loans[column].dtype==np.float64]
numeric_features_map = {}
for feature in numeric_features:
    numeric_features_map[feature] = feature+'_new'
for groupby in [['country', 'region'], ['country'], ['continent_region'], ['continent']]:
    replace_numeric_features_with_groupby_mean(groupby)
for feature in numeric_features:
    mean = kiva_loans[feature].mean()
    kiva_loans[feature].fillna(mean, inplace=True)

numeric_features = ['longitude', 'latitude']
numeric_features_map = {}
for feature in numeric_features:
    numeric_features_map[feature] = feature+'_new'
replace_numeric_features_with_groupby_mean(['country', 'region'], fixing_coordinates=True)

kiva_loans.drop(columns=['continent_region', 'borrower_female_count'], inplace=True)

del numeric_features_map, numeric_features
gc_collect()

report([column for column in kiva_loans.columns if kiva_loans[column].dtype==np.int64 or kiva_loans[column].dtype==np.float64])
start = t_time()

weights = dict(
    expected_years_of_schooling = 1,
    mean_years_of_schooling = 1,
    mean_years_of_schooling_pct = 1,
    gni_per_capta = 1,
    population_below_poverty_line = 2,
    life_expectancy = 2,
    happiness = 3,
    gpi = 3,
    hdi = 4
)

kiva_loans['bonstato_national'] = 0
weights_sum = 0

for indicator in indicators:
    kiva_loans['bonstato_national'] = kiva_loans['bonstato_national'] + weights[indicator]*kiva_loans[indicator]
    weights_sum += weights[indicator]

kiva_loans['bonstato_national'] = kiva_loans['bonstato_national']/weights_sum

kiva_loans.drop(columns=indicators, inplace=True)
gc_collect()

report(['bonstato_national'])
def plot_time_series(features, kind, series=['day', 'month', 'year', 'date']):
    
    start = t_time()
    
    if kind=='mean':
        aggreg=np.mean
    elif kind=='total':
        aggreg=np.sum
    
    vis_orig = kiva_loans[features+['posted_time']].copy()
    
    if 'missing_funds_pct' in features:
        vis_orig['missing_funds_pct'] = 1500*vis_orig['missing_funds_pct']

    if 'day' in series:
        vis = vis_orig.copy()
        vis['posted_time'] = vis['posted_time'].dt.day
        vis.groupby('posted_time').apply(aggreg).drop(columns=['posted_time']).plot(ax=plt.subplots(figsize=(15,3))[1])
        plt.xlabel('day of month')
        plt.ylabel(kind)

    if 'month' in series:
        vis = vis_orig.copy()
        vis['posted_time'] = vis['posted_time'].dt.month
        vis.groupby('posted_time').apply(aggreg).drop(columns=['posted_time']).plot(ax=plt.subplots(figsize=(15,3))[1])
        plt.xlabel('month')
        plt.ylabel(kind)

    if 'year' in series:
        vis = vis_orig.copy()
        vis['posted_time'] = vis['posted_time'].dt.year
        vis.groupby('posted_time').apply(aggreg).drop(columns=['posted_time']).plot(ax=plt.subplots(figsize=(15,3))[1])
        plt.xlabel('year')
        plt.ylabel(kind)

    if 'date' in series:
        vis = vis_orig.copy()
        vis['posted_time'] = vis['posted_time'].dt.date
        vis.groupby('posted_time').apply(aggreg).plot(ax=plt.subplots(figsize=(15,3))[1])
        plt.xlabel('date')
        plt.ylabel(kind)
    
    del vis_orig, vis
    gc_collect()
    timer(start)
plot_time_series(['loan_amount', 'funded_amount'], 'total', ['day', 'month', 'year'])
plot_time_series(['loan_amount', 'funded_amount'], 'mean', ['day', 'month', 'year'])
plot_time_series(['missing_funds', 'missing_funds_pct'], 'mean')
start = t_time()

corrs = kiva_loans.drop(columns=['latitude', 'longitude', 'missing_funds_pct']).corr()[['missing_funds']].drop(['missing_funds', 'bonstato_national'])
sns.heatmap(corrs.sort_values(by='missing_funds', ascending=False), ax=plt.subplots(figsize=(7,5))[1], annot=True)
plt.title('Correlations between loans\' numerical features and missing_funds')

corrs = kiva_loans.drop(columns=['latitude', 'longitude', 'missing_funds']).corr()[['missing_funds_pct']].drop(['missing_funds_pct', 'bonstato_national'])
sns.heatmap(corrs.sort_values(by='missing_funds_pct', ascending=False), ax=plt.subplots(figsize=(7,5))[1], annot=True)
plt.title('Correlations between loans\' numerical features and missing_funds_pct')

timer()
plot_time_series(['funded_amount_per_lender'], 'mean')
plot_time_series(['funded_amount_per_day'], 'mean', ['month', 'year', 'date'])
plot_time_series(['posting_delay_in_days', 'funding_delay_in_days', 'total_delay_in_days'], 'mean', ['year', 'date'])
plot_time_series(['lenders_experience_in_days_avg'], 'mean', ['year', 'date'])
plot_time_series(['borrower_count'], 'mean', ['year'])
plot_time_series(['borrower_female_pct'], 'mean', ['year'])
start = t_time()

for feature in ['currency', 'Field Partner Name']:
    for country in set(kiva_loans['country']):
        if country==np.nan:
            continue
        feature_values = list(set(kiva_loans[kiva_loans['country']==country][feature]))
        if len(feature_values) > 1 and np.nan not in feature_values:
            print('Example of country with more than 1 {}: {}'.format(feature, country))
            print('{} values: {}\n'.format(feature, '; '.join(feature_values)))
            break
    for feature_value in set(kiva_loans[feature]):
        if feature_value==np.nan:
            continue
        countries = list(set(kiva_loans[kiva_loans[feature]==feature_value]['country']))
        if len(countries) > 1 and not np.nan in countries and feature_value!='USD':
            print('Example of {} present in more than 1 country: {}'.format(feature, feature_value))
            print('Countries: {}\n'.format('; '.join(countries)))
            break

timer()
categorical_features = ['activity', 'sector', 'currency', 'Loan Theme Type', 'Field Partner Name', 'repayment_interval']
base_parameters = ['loan_amount', 'funded_amount', 'missing_funds']

def plot_categorical_feature(feature):
    
    start = t_time()
    
    vis = kiva_loans[base_parameters+[feature]].copy().groupby(feature).sum()
    vis['money_involved'] = vis['loan_amount'] + vis['funded_amount']
    vis['missing_funds_pct'] = vis['missing_funds']/vis['loan_amount']
    vis.drop(columns=base_parameters, inplace=True)
    
    n_categories = len(vis.index)
    
    for parameter in ['money_involved', 'missing_funds_pct']:
        if n_categories > 40:
            vis[[parameter]].sort_values(by=[parameter], ascending=False).head(20).plot(kind='bar', ax=plt.subplots(figsize=(15,1))[1])
            plt.title('top 20 {}: {}'.format(feature, parameter))
            plt.xlabel('')
        else:
            vis[[parameter]].sort_values(by=[parameter], ascending=False).plot(kind='bar', ax=plt.subplots(figsize=(15,1))[1])
            plt.title('{}: {}'.format(feature, parameter))
            plt.xlabel('')
        plt.show()
        plt.close('all')
    
    del vis
    gc_collect()
    timer(start)
for feature in categorical_features:
    plot_categorical_feature(feature)
start = t_time()

kiva_loans['borrower_female_count'] = kiva_loans['borrower_female_pct']*kiva_loans['borrower_count']
kiva_loans_by_country = rows_with_loans(kiva_loans).drop(columns=['latitude', 'longitude']).groupby('country').mean().reset_index()
kiva_loans_by_country['borrower_female_pct'] = kiva_loans_by_country['borrower_female_count']/kiva_loans_by_country['borrower_count']
kiva_loans_by_country['missing_funds_pct'] = kiva_loans_by_country['missing_funds']/kiva_loans_by_country['loan_amount']
kiva_loans_by_country.drop(columns=['borrower_female_count'], inplace=True)
kiva_loans.drop(columns=['borrower_female_count'], inplace=True)

corrs = kiva_loans_by_country.drop(columns=['MPI_national', 'MPI_regional']).corr()[['bonstato_national']].drop('bonstato_national')

pos_corrs = corrs[corrs['bonstato_national']>0][['bonstato_national']].sort_values(by='bonstato_national', ascending=False)
sns.heatmap(pos_corrs, ax=plt.subplots(figsize=(7,5))[1], annot=True)
plt.title('Positive correlations between loans\' numerical features and bonstato_national')

neg_corrs = corrs[corrs['bonstato_national']<0][['bonstato_national']].sort_values(by='bonstato_national', ascending=True)
sns.heatmap(neg_corrs, ax=plt.subplots(figsize=(7,5))[1], annot=True)
plt.title('Negative correlations between loans\' numerical features and bonstato_national')

del corrs, pos_corrs, neg_corrs
gc_collect()
timer()
start = t_time()
sns.jointplot(x='MPI_national', y='bonstato_national', data=kiva_loans_by_country, kind='reg')
timer()
def plot_categorical_feature_with_bonstato_threshold(feature, bonstato_threshold):
    
    start = t_time()
    
    n_loans_rich = kiva_loans[(kiva_loans['posted_time'].notna()) & (kiva_loans['bonstato_national']>bonstato_threshold)].shape[0]
    n_loans_poor = kiva_loans[(kiva_loans['posted_time'].notna()) & (kiva_loans['bonstato_national']<=bonstato_threshold)].shape[0]
    
    vis_rich = kiva_loans[(kiva_loans['posted_time'].notna()) & (kiva_loans['bonstato_national']>bonstato_threshold)].groupby(feature)[feature].count().sort_values(ascending=False)/n_loans_rich
    vis_poor = kiva_loans[(kiva_loans['posted_time'].notna()) & (kiva_loans['bonstato_national']<=bonstato_threshold)].groupby(feature)[feature].count().sort_values(ascending=False)/n_loans_poor
    
    top_rich = False
    if len(vis_rich.index) > 20:
        top_rich = True
        vis_rich = vis_rich.head(20)
    
    top_poor = False
    if len(vis_poor.index) > 20:
        top_poor = True
        vis_poor = vis_poor.head(20)
    
    vis_rich.plot(kind='bar', ax=plt.subplots(figsize=(15,1))[1], color='blue')
    plt.xlabel('')
    title = feature+' on rich countries'
    if top_rich:
        title = 'top 20 ' + title
    plt.title(title)
    
    vis_poor.plot(kind='bar', ax=plt.subplots(figsize=(15,1))[1], color='red')
    plt.xlabel('')
    title = feature+' on poor countries'
    if top_poor:
        title = 'top 20 ' + title
    plt.title(title)
    
    plt.show()
    plt.close('all')
    
    del vis_rich, vis_poor
    gc_collect()
    timer(start)
print('Number of countries helped by Kiva: {}'.format(kiva_loans_by_country.shape[0]))
start = t_time()
bonstato_threshold = kiva_loans_by_country.sort_values(by='bonstato_national', ascending=False).reset_index().loc[16:17, 'bonstato_national'].mean()
del kiva_loans_by_country
gc_collect()
print('Bonstato threshold: {}'.format(bonstato_threshold))
timer()
for feature in categorical_features:
    plot_categorical_feature_with_bonstato_threshold(feature, bonstato_threshold)
start = t_time()
kiva_loans = rows_with_loans(kiva_loans)
report([])
start = t_time()

def replace_loans(continent, lat_less_than=np.inf, lat_greater_than=-np.inf, lon_less_than=np.inf, lon_greater_than=-np.inf):
    kiva_subset = kiva_loans[(kiva_loans['continent']==continent)
                             & (kiva_loans['longitude']<=lon_less_than)
                             & (kiva_loans['longitude']>=lon_greater_than)
                             & (kiva_loans['latitude']<=lat_less_than)
                             & (kiva_loans['latitude']>=lat_greater_than)]
    
    kiva_not_subset = rows_with_coordinates(kiva_loans[kiva_loans['continent']==continent])
    kiva_not_subset = kiva_not_subset[[False if index in kiva_subset.index else True for index in kiva_not_subset.index]]
    
    for index in kiva_subset.index:
        country = kiva_subset.at[index, 'country']
        region = kiva_subset.at[index, 'region']
        
        if type(region) is float:
            longitude_median = kiva_not_subset[kiva_not_subset['country']==country]['longitude'].median()
            latitude_median = kiva_not_subset[kiva_not_subset['country']==country]['latitude'].median()
        else:
            longitude_median = kiva_not_subset[(kiva_not_subset['country']==country) & (kiva_not_subset['region']==region)]['longitude'].median()
            latitude_median = kiva_not_subset[(kiva_not_subset['country']==country) & (kiva_not_subset['region']==region)]['latitude'].median()
            if longitude_median==np.nan or latitude_median==np.nan:
                longitude_median = kiva_not_subset[kiva_not_subset['country']==country]['longitude'].median()
                latitude_median = kiva_not_subset[kiva_not_subset['country']==country]['latitude'].median()
        
        kiva_loans.at[index, 'longitude'] = longitude_median
        kiva_loans.at[index, 'latitude'] = latitude_median

def fix_coordinates(country, region, longitude, latitude):
    kiva_loans.loc[(kiva_loans['country']==country) & (kiva_loans['region']==region), 'longitude'] = longitude
    kiva_loans.loc[(kiva_loans['country']==country) & (kiva_loans['region']==region), 'latitude'] = latitude

fix_coordinates('Mexico', 'Amatepec', -100.3862514, 18.6996269)
fix_coordinates('Mexico', 'Hacienda Vieja', -101.8256992, 18.7291864)
fix_coordinates('Mexico', 'Iztacalco', -99.1315041, 19.3990446)
fix_coordinates('Ecuador', 'Buenos Aires', -79.69902, -0.081719)
fix_coordinates('Peru', 'La Tuna San José Del Alto', -79.0871618, -5.4345415)
fix_coordinates('Peru', 'Icamanche', -77.1278655, -12.0262676)
fix_coordinates('Peru', 'Flor De Café Las Pirias', -77.1278655, -12.0262676)
fix_coordinates('Peru', 'El Pedregal San José De Lourdes/Los Alpes San José De Lourdes', -78.9004218, -5.0994225)
fix_coordinates('Peru', 'La Tuna San José Del Alto', -79.0871618, -5.4345415)
fix_coordinates('Peru', 'José De San Martín Tabaconas', -77.1278661, -12.0255262)
fix_coordinates('Bolivia', 'Casamaya', -66.1655564, -17.3769575)
fix_coordinates('Bolivia', 'Mucuña', -68.0459891, -16.2256556)
fix_coordinates('The Democratic Republic of the Congo', 'Kavumu Territoire De Kabre Sud Kivu', 28.7419053, -2.256716)
fix_coordinates('Mali', 'Fala Sounkoro', -7.0587548, 11.5500206)
fix_coordinates('Haiti', 'Croix Des Bouquets', -72.2396615, 18.578869)
fix_coordinates('Mexico', 'Tierra Amarilla', -92.8900215, 18.1178861)
fix_coordinates('Ecuador', 'San Francisco', -78.570625, -0.1862504)
fix_coordinates('Peru', 'Santa Fe', -77.143164, -12.0150627)
fix_coordinates('Kenya', 'Mogadishu Somalia', 36.8604688, -1.3071561)
fix_coordinates('Mali', 'Yolla', -11.6420878, 14.2833534)
fix_coordinates('Georgia', 'Puti', 41.6046806, 42.138998)

replace_loans('Americas', lon_greater_than=-38)
replace_loans('Oceania', lat_greater_than=30)
replace_loans('Asia', lon_less_than=0)
replace_loans('Asia', lon_less_than=40, lat_less_than=0)
replace_loans('Africa', lon_less_than=-18)
replace_loans('Africa', lon_greater_than=50)
replace_loans('Africa', lat_greater_than=20)

kiva_loans.loc[(kiva_loans['country']=='Mali') & (kiva_loans['region']=='Congo'), 'country'] = 'Congo'
kiva_loans.loc[(kiva_loans['country']=='Congo') & (kiva_loans['region']=='Congo'), 'region'] = 'Mali'

timer()
data_quality_report()
kiva_loans.info()
start = t_time()
kiva_loans = pd.get_dummies(data=kiva_loans, dummy_na=True, drop_first=True, columns=categorical_features)
report()
start = t_time()

kiva_loans['borrower_female_count'] = kiva_loans['borrower_female_pct']*kiva_loans['borrower_count']

kiva_countries = kiva_loans.drop(columns=['latitude', 'longitude']).groupby(['country']).mean().reset_index()
kiva_countries['borrower_female_pct'] = kiva_countries['borrower_female_count']/kiva_countries['borrower_count']
kiva_countries['missing_funds_pct'] = kiva_countries['missing_funds']/kiva_countries['loan_amount']
kiva_countries.drop(columns=['borrower_female_count'], inplace=True)

kiva_regions = rows_with_countries_and_regions(kiva_loans).groupby(['country', 'region']).mean().reset_index()
kiva_regions['borrower_female_pct'] = kiva_regions['borrower_female_count']/kiva_regions['borrower_count']
kiva_regions['missing_funds_pct'] = kiva_regions['missing_funds']/kiva_regions['loan_amount']
kiva_regions.drop(columns=['borrower_female_count'], inplace=True)

del kiva_loans

print('Train data size: {}'.format(kiva_countries.shape[0]))
print('Test data size: {}'.format(kiva_regions.shape[0]))

gc_collect()
timer()
class BonstatoRegressor:
    
    def train_add_model(self, model, n_countries, X, y):
        start=t_time()
        score = -1/np.array(cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=n_countries)).mean()
        model.fit(X,y)
        name = str(model).split('(')[0]
        training = t_time()-start
        print('{}\n\tscore: {:.3f}\n\ttraining(s): {:.3f}\n'.format(name, score, training))
        model_score = pd.DataFrame({'name':[name], 'score':[score], 'model':[model], 'training(s)':[training]})
        return self.model_scores_.append(model_score, ignore_index=True)
    
    def __init__(self):
        self.model_scores_ = pd.DataFrame({'name':[], 'model':[], 'score':[], 'training(s)':[]})
        self.models = [
            AdaBoostRegressor(random_state=42),
            BaggingRegressor(n_estimators=30, random_state=42),
            RandomForestRegressor(n_estimators=50, random_state=42),
            ExtraTreesRegressor(n_estimators=30, random_state=42),
            GradientBoostingRegressor(random_state=42),
            DecisionTreeRegressor(random_state=42),
            ExtraTreeRegressor(random_state=42),
            Lasso(normalize=True, random_state=42),
            Ridge(random_state=42),
            PassiveAggressiveRegressor(max_iter=1500, random_state=42),
            LinearRegression(),
            TheilSenRegressor(random_state=42),
            KNeighborsRegressor(),
            KernelRidge(),
            GaussianProcessRegressor(random_state=42)
        ]
    
    def fit(self, n_countries, X, y):
        for model in self.models:
            self.model_scores_ = self.train_add_model(model, n_countries, X, y)
    
    def predict(self, X):
        raw_predictions = pd.DataFrame()
        raw_predictions_weighted = pd.DataFrame()
        result = pd.DataFrame()
        for i in range(self.model_scores_.shape[0]):
            name = self.model_scores_.at[i, 'name']
            score = self.model_scores_.at[i, 'score']
            model = self.model_scores_.at[i, 'model']
            raw_predictions[name] = model.predict(X)
            raw_predictions_weighted[name] = score * raw_predictions[name]

        transpose = raw_predictions_weighted.transpose()
        predictions = []
        for column in transpose.columns:
            predictions.append(transpose[column].sum())
        result['prediction'] = np.array(predictions).transpose()/self.model_scores_['score'].sum()

        result['uncertainty'] = 0
        for name in set(self.model_scores_['name']):
            score = self.model_scores_.loc[self.model_scores_['name']==name, 'score'].values[0]
            result['uncertainty'] = result['uncertainty'] + score*np.absolute(raw_predictions[name] - result['prediction'])
        result['uncertainty'] = result['uncertainty']/(self.model_scores_['score'].sum())
        result['uncertainty'] = result['uncertainty']/result['prediction']
        result.loc[result['uncertainty']>1, 'uncertainty'] = 1
        return result
start = t_time()

bonstato_regressor = BonstatoRegressor()

bonstato_regressor.fit(87, kiva_countries.drop(columns=['country',
                                                        'bonstato_national',
                                                        'MPI_national',
                                                        'MPI_regional']),
                       kiva_countries['bonstato_national'])

result = bonstato_regressor.predict(kiva_regions.drop(columns=['country',
                                                               'region',
                                                               'longitude',
                                                               'latitude',
                                                               'bonstato_national',
                                                               'MPI_national',
                                                               'MPI_regional']))

del kiva_countries
gc_collect()

timer()

bonstato_regressor.model_scores_.drop(columns=['model']).sort_values(by='score', ascending=False).reset_index(drop=True)
start = t_time()
sns.lmplot(x='prediction', y='uncertainty', data=result, size=5, aspect=3)
timer()
start = t_time()
sns.distplot(result['uncertainty'], kde=False, bins=200, ax=plt.subplots(figsize=(15,5))[1])
timer()
result.describe()[['uncertainty']]
start = t_time()

kiva_regions['bonstato_regional'] = (1-result['uncertainty'])*result['prediction'] + result['uncertainty']*kiva_regions['bonstato_national']
kiva_regions['uncertainty'] = result['uncertainty']

output = kiva_regions[['country',
                       'region',
                       'longitude',
                       'latitude',
                       'bonstato_regional',
                       'uncertainty']].sort_values(by='bonstato_regional').reset_index(drop=True)

del result, bonstato_regressor
gc_collect()
timer()
output.head(4)
start = t_time()
output.to_csv('countries_regions_sorted_by_bonstato.csv', index=False)
output.sort_values(by=['country', 'bonstato_regional']).to_csv('countries_regions_sorted_by_country.csv', index=False)
del output
gc_collect()
timer()
start = t_time()
print('Correlation between bonstato_national and bonstato_regional: {:.3f}\n'.format(kiva_regions[['bonstato_regional',
                                                                                                   'bonstato_national']].corr().at['bonstato_regional',
                                                                                                                                   'bonstato_national']))
sns.lmplot(x='bonstato_national', y='bonstato_regional', data=kiva_regions, size=5, aspect=3)
plt.plot([], [], label='data')
plt.plot([0,1], [0,1], color='r', label='identity')
plt.legend(shadow=True)
timer()
start = t_time()
sns.jointplot(x='MPI_regional', y='bonstato_regional', data=kiva_regions, kind='reg')
timer()
N_CLUSTERS = 870

start = t_time()

kiva_regions = rows_with_coordinates(kiva_regions).copy()

kiva_regions['cluster'] = KMeans(N_CLUSTERS, random_state=42).fit_predict(kiva_regions[['longitude', 'latitude']])
kiva_regions_clusters_groupby = kiva_regions[['country', 'region', 'longitude', 'latitude', 'bonstato_regional', 'uncertainty', 'cluster']].groupby('cluster')
kiva_regions_clusters = kiva_regions_clusters_groupby.mean().reset_index()
for column in ['longitude', 'latitude', 'bonstato_regional', 'uncertainty']:
    kiva_regions_clusters.rename(columns={column: column+'_mean'}, inplace=True)
    kiva_regions_clusters[column+'_std'] = kiva_regions_clusters_groupby[column].apply(np.std)
kiva_regions_clusters['countries'] = kiva_regions_clusters_groupby['country'].apply(join_names)
kiva_regions_clusters['regions'] = kiva_regions_clusters_groupby['region'].apply(join_names)

kiva_regions_clusters = kiva_regions_clusters.reset_index()[['cluster',
                                                           'countries',
                                                           'regions',
                                                           'longitude_mean',
                                                           'longitude_std',
                                                           'latitude_mean',
                                                           'latitude_std',
                                                           'bonstato_regional_mean',
                                                           'bonstato_regional_std',
                                                           'uncertainty_mean',
                                                           'uncertainty_std']].sort_values(by='bonstato_regional_mean', ascending=False)

def truncate(obj):
    if len(obj) > 56:
        obj = obj[:53]+'...'
    return obj

data = [ dict(
    type = 'scattergeo',
    lon = kiva_regions_clusters['longitude_mean'],
    lat = kiva_regions_clusters['latitude_mean'],
    text = '<br><b>Cluster</b><br>'+kiva_regions_clusters['cluster'].apply(str)
        +'<br><br><b>Countries</b><br>'+kiva_regions_clusters['countries'].apply(truncate)
        +'<br><br><b>Regions</b><br>'+kiva_regions_clusters['regions'].apply(truncate)
        +'<br><br><b>Mean Bonstato</b><br>'+kiva_regions_clusters['bonstato_regional_mean'].apply(str)
        +'<br><br><b>Std. Bonstato</b><br>'+kiva_regions_clusters['bonstato_regional_std'].apply(str)
        +'<br><br><b>Mean Uncertainty</b><br>'+kiva_regions_clusters['uncertainty_mean'].apply(str)
        +'<br><br><b>Std. Uncertainty</b><br>'+kiva_regions_clusters['uncertainty_std'].apply(str),
    marker = dict(
        size = 10+10*(kiva_regions_clusters['longitude_std']+kiva_regions_clusters['latitude_std']),
        line = dict(
            width=0.1
        ),
        reversescale = True,
        colorscale='Portland',
        cmin = kiva_regions_clusters['bonstato_regional_mean'].min(),
        color = kiva_regions_clusters['bonstato_regional_mean'],
        cmax = kiva_regions_clusters['bonstato_regional_mean'].max(),
        opacity = 1,
        colorbar=dict(
            title="Bonstato"
        )
    )
)]

layout = dict(
    title = 'Bonstato by clusters',
    geo = dict(
        showcountries = True
    )
)

fig = dict(data=data, layout=layout)
py.iplot(fig)

kiva_regions_clusters = kiva_regions_clusters.sort_values(by='bonstato_regional_mean').reset_index(drop=True)
kiva_regions_clusters.to_csv('clusters_sorted_by_bonstato.csv', index=False)
kiva_regions_clusters.sort_values(by=['countries', 'bonstato_regional_mean']).to_csv('clusters_sorted_by_country.csv', index=False)
kiva_regions_clusters.sort_values(by=['cluster']).to_csv('clusters_sorted_by_cluster.csv', index=False)

del kiva_regions_clusters_groupby, data, layout, fig
gc_collect()

timer()
MIN_CLUSTERS = 8

def plot_contour_map(data, lon_col, lat_col, var_col, cmap, extent, lon_center, lat_center,
                     filename=None, weight_col=None, label_col=None, title=None, display=True):
    
    lon_min, lon_max, lat_min, lat_max = extent
    
    plt.subplots(figsize=(20, 20))

    m = Basemap(projection='merc', resolution='l',llcrnrlon=lon_min,llcrnrlat=lat_min,urcrnrlon=lon_max,urcrnrlat=lat_max,
                lon_0=(lon_max+lon_min)/2,lat_0=(lat_max+lat_min)/2)
    m.drawmapboundary(fill_color='#ccccff')
    m.drawcountries()
    m.drawcoastlines()
    
    # draw parallels
    parallels = np.arange(-90.,90,5.)
    m.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)

    # draw meridians
    meridians = np.arange(-360.,360.,5.)
    m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)
    
    coords_m = np.empty((data.shape[0], 2))
    coords_m[:, 0], coords_m[:, 1] = m(data[lon_col].values, data[lat_col].values)
    nn = NearestNeighbors(n_neighbors = min(5, data.shape[0]))
    nn.fit(coords_m)
    
    def get_surface_value(x, y):
        [dists], [indexes] = nn.kneighbors([[x, y]])
        sum_weight = 0
        sum_value = 0
        for dist, index in zip(dists, indexes):
            if dist==0:
                return data.at[index, var_col]
            if weight_col is None:
                raw_weight = 1
            else:
                raw_weight = data.at[index, weight_col]
            weight = raw_weight/dist
            sum_value += weight*data.at[index, var_col]
            sum_weight += weight
        return sum_value/sum_weight
    
    lon_length = lon_max-lon_min
    lat_length = lat_max-lat_min
    n_lon_bins = int(round(4*lon_length))
    n_lat_bins = int(round(4*lat_length))
    
    lons, lats = m.makegrid(n_lon_bins, n_lat_bins)
    X, Y = m(lons, lats)
    Z = np.empty(X.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i,j] = get_surface_value(X[i,j], Y[i,j])
    
    ocean_masked_Z = maskoceans(lons,lats,Z)
    
    n_levels = 2*max(n_lon_bins, n_lat_bins)
    
    cs1 = m.contour(lons, lats, ocean_masked_Z, n_levels, colors='k', latlon=True, linewidths=0)
    cs2 = m.contourf(lons, lats, ocean_masked_Z, cs1.levels, extend='both', cmap=cmap, latlon=True)
    cbar = m.colorbar(cs2)
    cbar.set_label('Bonstato')
    
    if not label_col is None:
        for x,lon,y,lat,l in zip(coords_m[:, 0], data[lon_col], coords_m[:, 1], data[lat_col], data[label_col].apply(str)):
            if lon_min<lon and lon<lon_max and lat_min<lat and lat<lat_max:
                plt.scatter(x, y, marker='o', color='black')
                plt.text(x, y, ' '+l)
    
    x_text, y_text = m(lon_center, lat_center)
    text = plt.text(x_text, y_text, title, fontsize=25, color='white', fontweight='bold')
    text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='black'), path_effects.Normal()])
    
    if not title is None:
        plt.title(title, fontsize=30)
    if not filename is None:
        plt.savefig(filename)
    if not display:
        plt.close()
    
    del m, coords_m, nn, lons, lats, X, Y, Z, ocean_masked_Z, cs1, cs2
    gc_collect()
start = t_time()

lon_lat_std_mean = kiva_regions_clusters['longitude_std'].mean()+kiva_regions_clusters['latitude_std'].mean()    

for country in set(kiva_regions['country']):
    
    # these countries only have 1 contour level thus cannot be plotted
    if country in ['Afghanistan', 'Congo', 'Saint Vincent and the Grenadines']:
        continue
    
    subset_region = kiva_regions[kiva_regions['country']==country]
    lon_min, lon_max, lat_min, lat_max = (subset_region['longitude'].min()-1,
                                          subset_region['longitude'].max()+1,
                                          subset_region['latitude'].min()-1,
                                          subset_region['latitude'].max()+1)
    
    subset = kiva_regions_clusters[(lon_min-2 < kiva_regions_clusters['longitude_mean'])
                                   & (kiva_regions_clusters['longitude_mean'] < lon_max+2)
                                   & (lat_min-2 < kiva_regions_clusters['latitude_mean'])
                                   & (kiva_regions_clusters['latitude_mean'] < lat_max+2)].reset_index(drop=True)
    
    if subset.shape[0] >= MIN_CLUSTERS:
    
        subset['weight'] = (1-subset['uncertainty_mean'])*(subset['longitude_std']+subset['latitude_std']+lon_lat_std_mean)

        filename=country.replace('\'', '').replace('(', '-').replace(')', '')+'.png'

        plot_contour_map(data=subset, lon_col='longitude_mean', lat_col='latitude_mean', var_col='bonstato_regional_mean',
                         cmap=plt.cm.jet_r, extent=(lon_min, lon_max, lat_min, lat_max), lon_center=subset_region['longitude'].mean(),
                         lat_center=subset_region['latitude'].mean(), filename=filename,
                         weight_col='weight', label_col='cluster', title=country, display=(country in ['India', 'Mexico']))
    
    del subset_region, subset
    gc_collect()

del kiva_regions, kiva_regions_clusters
gc_collect()

z = zipfile.ZipFile('interpolations.zip', 'w')
for file in os.listdir():
    if file.endswith('.png'):
        z.write(file)
        os.remove(file)
z.close()

timer()
raw_timer()