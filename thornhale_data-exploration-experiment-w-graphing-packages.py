import pandas as pd

import holoviews as hv

import seaborn as sb

import matplotlib as mpl

import matplotlib.pyplot as plt

import numpy as np

import scipy.stats as stats

import pylab

%matplotlib inline
train_df = pd.read_csv('./train/train_2016.csv')

property_df = pd.read_csv('./train/properties_2016.csv')
train_df.head(5)
property_df.head(5)
len(property_df.columns)
property_df.columns
property_df.columns = ['parcelid', 

'ac_id', 

'id_arci_style',

'area_basement', 

'num_bathroom', 

'num_bedroom', 

'id_build_class',

'id_build_quality', 

'calculatedbathnbr', 

'id_decktype',

'area_first_floor', 

'area_total_calc',

'area_fin_living', 

'area_fin_perim_living', 

'area_fin_total_area',

'area_first_floor_2', 

'area_base', 

'fips', 

'num_fire',

'num_fullbath', 

'num_garagecar', 

'area_garage', 

'has_spa',

'id_heating_system_id', 

'latitude', 

'longitude', 

'area_lotsize',

'num_pool', 

'area_pool_total', 

'id_spa_tub', 

'id_pool_spa_hottub', 

'id_pool_no_hottub',

'id_zone_county_landusecode', 

'id_zone_landuse',

'zone_property', 

'rawcensustractandblock', 

'region_city',

'region_county', 

'region_neighborhood',

'region_zip', 

'num_room',

'id_storytype', 

'num_3_4_bath', 

'typeconstructiontypeid',

'num_unit', 

'area_patio_yd', 

'area_shed_yd', 

'year_built',

'num_stories', 

'has_fireplace', 

'assessed_home_value',

'assessed_parcel_value', 

'assessmentyear', 

'landtaxvaluedollarcnt',

'tax_amount', 

'tax_is_delinquent', 

'tax_delinquency_year',

'censustractandblock']
missing_df = property_df.isnull().sum(axis=0).reset_index()

missing_df.columns = ['column_name', 'missing_count']

filled_df = property_df.notnull().sum(axis=0).reset_index()

filled_df.columns = ['column_name', 'filled_count']

merged_df = pd.merge(missing_df, filled_df, on=['column_name'])

merged_df['fraction_filled'] = merged_df['filled_count']/(merged_df['missing_count'] + merged_df['filled_count'])*100

merged_df = merged_df.loc[merged_df['missing_count']>0]

merged_df = merged_df.sort_values(by='missing_count', ascending=True)

merged_df.head()
fig = plt.figure(figsize=(20, 16))

sn_plot = sb.barplot(x='fraction_filled', y='column_name', data=merged_df)

sn_plot.set_xlabel('Percent', fontsize = 25)

sn_plot.set_ylabel('Feature', fontsize = 25)

sn_plot.set_title('Feature Completeness', fontsize=50)
hv.notebook_extension('bokeh')

frequencies, edges = np.histogram(train_df['logerror'], 500)

hv.Histogram(frequencies, edges)
stats.probplot(train_df['logerror'], dist='norm', plot=pylab)
full_train_df = pd.merge(train_df, property_df, on=['parcelid'])
non_categorical_features_df = full_train_df[[

    'logerror',

    'transactiondate',

    'area_basement', 

    'num_bathroom', 

    'num_bedroom', 

    'calculatedbathnbr', 

    'area_first_floor', 

    'area_total_calc',

    'area_fin_living', 

    'area_fin_perim_living', 

    'area_fin_total_area',

    'area_first_floor_2', 

    'area_base', 

    'num_fire',

    'num_fullbath', 

    'num_garagecar', 

    'area_garage', 

    'has_spa',

    'latitude', 

    'longitude', 

    'area_lotsize',

    'num_pool', 

    'area_pool_total', 

    'num_room',

    'num_3_4_bath', 

    'num_unit', 

    'area_patio_yd', 

    'area_shed_yd', 

    'year_built',

    'num_stories', 

    'has_fireplace', 

    'assessed_home_value',

    'assessed_parcel_value', 

    'assessmentyear', 

    'landtaxvaluedollarcnt',

    'tax_amount', 

    'tax_is_delinquent', 

    'tax_delinquency_year',

]]
cor_mat = non_categorical_features_df.corr(method='spearman')

sb.heatmap(cor_mat)
sparse_features_df = full_train_df[[

    'logerror',

    'transactiondate',

    'area_basement', 

    'calculatedbathnbr', 

    'num_fire',

    'num_fullbath', 

    'area_garage', 

    'has_spa',

    'num_pool', 

    'area_pool_total', 

    'num_3_4_bath', 

    'area_patio_yd', 

    'area_shed_yd', 

    'has_fireplace', 

    'tax_is_delinquent', 

    'tax_delinquency_year',

    'id_spa_tub',

    'id_pool_spa_hottub'

]].copy()
sparse_features_df.describe()
sparse_features_df['num_fire'].fillna(0, inplace=True)

fig = sb.violinplot(y='logerror', x='num_fire', data=sparse_features_df)

fig.axes.set_ylim(-0.35,0.35)

plt.show(fig)
sparse_features_df['num_3_4_bath'].fillna(0, inplace=True)

fig = sb.violinplot(y='logerror', x='num_3_4_bath', data=sparse_features_df)

fig.axes.set_ylim(-0.35,0.35)

plt.show(fig)
sparse_features_df['has_spa'].fillna(False, inplace=True)

fig = sb.violinplot(y='logerror', x='has_spa', data=sparse_features_df)

fig.axes.set_ylim(-0.25,0.25)
sparse_features_df['num_pool'].fillna(0, inplace=True)

fig = sb.violinplot(y='logerror', x='num_pool', data=sparse_features_df, scale="width")

fig.axes.set_ylim(-0.25,0.25)

plt.show(fig)
sparse_features_df['has_fireplace'].fillna(False, inplace=True)

fig = sb.violinplot(y='logerror', x='has_fireplace', data=sparse_features_df)

fig.axes.set_ylim(-0.25,0.25)

plt.show(fig)
sparse_features_df['id_spa_tub'].fillna(False, inplace=True)

fig = sb.violinplot(y='logerror', x='id_spa_tub', data=sparse_features_df)

fig.axes.set_ylim(-0.25,0.25)

plt.show(fig)
sparse_features_df['id_pool_spa_hottub'].fillna(False, inplace=True)

fig = sb.violinplot(y='logerror', x='id_pool_spa_hottub', data=sparse_features_df)

fig.axes.set_ylim(-0.25,0.25)

plt.show(fig)