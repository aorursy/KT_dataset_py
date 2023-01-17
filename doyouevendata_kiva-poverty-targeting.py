import numpy as np # linear algebra
from numpy import log10, ceil, ones
from numpy.linalg import inv 
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # prettier graphs
import matplotlib.pyplot as plt # need dis too
%matplotlib inline 
from IPython.display import HTML # for da youtube memes
import itertools # let's me iterate stuff
from datetime import datetime # to work with dates
import geopandas as gpd
from scipy import stats
from fuzzywuzzy import process
from shapely.geometry import Point, Polygon
import shapely.speedups
shapely.speedups.enable()
import fiona 
from time import gmtime, strftime
from shapely.ops import cascaded_union
import gc

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

sns.set_style('darkgrid') # looks cool, man
import os

import warnings
warnings.filterwarnings('ignore')
# a helpful buddy
# https://stackoverflow.com/questions/30942577/seaborn-correlation-coefficient-on-pairgrid
def corrfunc(x, y, **kws):
    r, _ = stats.pearsonr(x, y)
    ax = plt.gca()
    ax.annotate("r = {:.2f}".format(r),
                xy=(.1, .9), xycoords=ax.transAxes)
# read some files...

# read in MPI
df_mpi_ntl = pd.read_csv("../input/mpi/MPI_national.csv")
df_mpi_subntl = pd.read_csv("../input/mpi/MPI_subnational.csv")

# read in kiva data
df_kv_loans = pd.read_csv("../input/data-science-for-good-kiva-crowdfunding/kiva_loans.csv")
df_kv_theme = pd.read_csv("../input/data-science-for-good-kiva-crowdfunding/loan_theme_ids.csv")
df_kv_theme_rgn = pd.read_csv("../input/data-science-for-good-kiva-crowdfunding/loan_themes_by_region.csv")

# read in kiva enhanced data
df_kiv_loc = pd.read_csv("../input/kiva-challenge-coordinates/kiva_locations.csv", sep='\t', error_bad_lines=False)

# read in cato data
df_cato = pd.read_csv("../input/cato-2017-human-freedom-index/cato_2017_human_freedom_index.csv")

# read in world bank findex and rural population data
df_wb_findex = pd.read_csv("../input/findex-world-bank/FINDEXData.csv")
df_wb_rural = pd.read_csv("../input/world-bank-rural-population/rural_pop.csv")
df_wb_cons = pd.read_csv("../input/world-bank-household-final-consumption-expenditure/wb_cons.csv")
df_wb_pop = pd.read_csv("../input/world-population/WorldPopulation.csv")

# read in kenya poverty metrics
df_pov_ken_food = pd.read_csv("../input/kenya-poverty-metrics-by-district/food_poverty_est.csv")
df_pov_ken_ovrl = pd.read_csv("../input/kenya-poverty-metrics-by-district/overall_poverty_est.csv")
df_pov_ken_hrdc = pd.read_csv("../input/kenya-poverty-metrics-by-district/hardcore_poverty_est.csv")
df_pov_ken_crdt = pd.read_csv("../input/kenya-poverty-metrics-by-district/credit_access.csv")

# read in india poverty metrics
df_ind_pov = pd.read_csv("../input/india-consumption-data-by-state/ind_poverty.csv")
df_ind_cons = pd.read_csv("../input/india-consumption-data-by-state-201112/ind_cons.csv")

# read in additional country stats
df_add_cntry = pd.read_csv("../input/additional-kiva-snapshot/country_stats.csv")

# read in philippines demographics
df_demo_phl = pd.read_csv("../input/kiva-phillipines-regional-info/phil_regions.csv")

# read in rwanda district banking info
df_rwa_bank = pd.read_csv("../input/rwanda-financial-exclusion-data/rwa_exclusion.csv")
LT = pd.read_csv("../input/data-science-for-good-kiva-crowdfunding/loan_themes_by_region.csv") #.set_index([''])
MPI = pd.read_csv("../input/mpi/MPI_national.csv")[['ISO','MPI Urban','MPI Rural']].set_index("ISO")
LT = LT.join(MPI,how='left',on="ISO")[['Partner ID','Field Partner Name','ISO','MPI Rural','MPI Urban','rural_pct','amount', 'mpi_region']].dropna()

LT['rural_pct'] /= 100
#~ Compute the MPI Score for each loan theme
LT['MPI Score'] = LT['rural_pct']*LT['MPI Rural'] + (1-LT['rural_pct'])*LT['MPI Urban']

LT[LT['ISO'] == 'MOZ'][['Partner ID', 'ISO', 'MPI Rural', 'MPI Urban', 'rural_pct', 'MPI Score']].drop_duplicates()
df_kiva_rural = df_kv_theme_rgn[['Partner ID', 'rural_pct']].drop_duplicates()
df_kiva_rural['populated'] = np.where(df_kiva_rural['rural_pct'].isnull(), 'No', 'Yes')
df_kiva_rural['populated'].value_counts()
df_mpi_subntl[df_mpi_subntl['ISO country code'] == 'MOZ'][['ISO country code', 'World region', 'MPI National']].head(1)
df_kv_theme_rgn[df_kv_theme_rgn['country'] == 'Mozambique'][['Partner ID', 'rural_pct']].merge(df_kv_theme_rgn[df_kv_theme_rgn['country'] == 'Mozambique'][['Partner ID', 'number', 'amount']].groupby('Partner ID').sum().reset_index(), on='Partner ID').drop_duplicates()
df_out = df_mpi_ntl[['ISO', 'MPI Rural', 'MPI Urban']].merge(df_wb_rural[['ISO', '2016']], on='ISO')
df_out = df_out.merge(df_mpi_subntl[['ISO country code', 'MPI National']].drop_duplicates(), left_on='ISO', right_on='ISO country code')
df_out['MPI National Calculated'] = df_out['2016'] / 100 * df_out['MPI Rural'] + (100 - df_out['2016']) / 100 * df_out['MPI Urban']
df_out['MPI National Difference'] = df_out['MPI National'] - df_out['MPI National Calculated']
df_out.drop(columns=['ISO country code', '2016'], inplace=True)
#df_out[df_out['ISO'] == 'MOZ']
df_out[df_out['ISO'].isin(['MOZ', 'RWA', 'AFG', 'GNB', 'BDI'])]
HTML('<img style="margin: 0px 20px" align=left src=https://www.doyouevendata.com/wp-content/uploads/2018/03/attn.png>Current Kiva methodology done for National MPI causes partners to drop from comparison.  We can attempt to keep these partners in the evaluation by applying the general rural percentage to the country to them; this can be done by taking the MPI provided MPI National from the Sub-National file, or if missing, calculating it ourselves for a country based on World Bank data.  This treats field partners as if they loan equally throughout the country; which they likely do not; although this may be preferrable to simply dropping them out of analysis entirely.')
plt.figure(figsize=(12,6))
sns.distplot(df_kiva_rural[~df_kiva_rural['rural_pct'].isnull()]['rural_pct'], bins=30)
plt.show()
plt.figure(figsize=(15,10))
plotSeries = df_kiva_rural['rural_pct'].value_counts().to_frame().reset_index()
o = plotSeries['index']
ax = sns.barplot(data=plotSeries, x='rural_pct', y='index', color='c', orient='h', order=o)
ax.set_title('Number of Field Partners with Rural Percentage', fontsize=15)
ax.set_ylabel('Rural Percentage')
ax.set_xlabel('Number of Field Partners')
plt.show()
HTML('<img style="margin: 0px 20px" align=left src=https://www.doyouevendata.com/wp-content/uploads/2018/03/attn.png>If Kiva wants to know this value, it would likely be best that require it as a loan level attribute.  This would resolve all issues going forward.  All field partners would have values, the numbers produced would be accurate for all partners, and the proper percentage rural could be calculated for any span of time.')
HTML('<img style="margin: 0px 20px" align=left src=https://www.doyouevendata.com/wp-content/uploads/2018/03/attn.png>It feels like the data provided has been produced from their application database; does Kiva have a separate reporting database?  A reporting database/data warehouse could accurately capture the application data as it is at a snapshot in time and track slowly changing dimension values along with it, so things like the proper MPI value is associated with a set of loans at a given point in time.')
ISO = 'MOZ'
MPIsubnat = pd.read_csv("../input/mpi/MPI_subnational.csv")[['Country', 'Sub-national region', 'World region', 'MPI National', 'MPI Regional']]
# Create new column LocationName that concatenates the columns Country and Sub-national region
MPIsubnat['LocationName'] = MPIsubnat[['Sub-national region', 'Country']].apply(lambda x: ', '.join(x), axis=1)

LTsubnat = pd.read_csv("../input/data-science-for-good-kiva-crowdfunding/loan_themes_by_region.csv")[['Partner ID', 'Loan Theme ID', 'region', 'mpi_region', 'ISO', 'number', 'amount', 'LocationName', 'names']]

# Merge dataframes
LTsubnat = LTsubnat.merge(MPIsubnat, left_on='mpi_region', right_on='LocationName', suffixes=('_LTsubnat', '_mpi'))[['Partner ID', 'Loan Theme ID', 'Country', 'ISO', 'mpi_region', 'MPI Regional', 'number', 'amount']]

#~ Get total volume and average MPI Regional Score for each partner loan theme
LS = LTsubnat.groupby(['Partner ID', 'Loan Theme ID', 'Country', 'ISO']).agg({'MPI Regional': np.mean, 'amount': np.sum, 'number': np.sum})
#~ Get a volume-weighted average of partners loanthemes.
weighted_avg_LTsubnat = lambda df: np.average(df['MPI Regional'], weights=df['amount'])
#~ and get weighted average for partners. 
MPI_regional_scores = LS.groupby(level=['Partner ID', 'ISO']).apply(weighted_avg_LTsubnat)
MPI_regional_scores = MPI_regional_scores.to_frame()
MPI_regional_scores.reset_index(level=1, inplace=True)
MPI_regional_scores = MPI_regional_scores.rename(index=str, columns={0: 'MPI Score'})

MPI_regional_scores[MPI_regional_scores['ISO'] == ISO].reset_index()
df_mpi_subntl[df_mpi_subntl['ISO country code'] == 'MOZ'][['ISO country code', 'Sub-national region', 'MPI Regional']].sort_values('Sub-national region')
df_kv_theme_rgn[(df_kv_theme_rgn['ISO'] == 'MOZ') & (df_kv_theme_rgn['Partner ID'] == 23)].groupby(['region', 'mpi_region'])[['number', 'amount']].sum().sort_values('amount', ascending=False)
HTML('<img style="margin: 0px 20px" align=left src=https://www.doyouevendata.com/wp-content/uploads/2018/03/attn.png>Kiva needs to update the accuracy of its current mpi region assignment.  The google API others have leveraged to find points seems to work reasonably well, as does the points in polygon geospatial approach.  It would probably work better at scale with spatial join functionality vs. my crude brute force code.  Kiva could also update their application to require some kind of valid google location point as well.')
# MOZ spatial projection
epsg = '42106'

### POINTS ###
# loan theme geodataframe
gdf_loan_theme = df_kv_theme_rgn[['Partner ID', 'region', 'mpi_region', 'ISO', 'number', 'amount']].groupby(['Partner ID', 'region', 'mpi_region', 'ISO']).sum().reset_index().merge(df_kiv_loc, how='left', on='region')
gdf_loan_theme['geometry'] = gdf_loan_theme.apply(lambda row: Point(row['lng'], row['lat']), axis=1)
gdf_loan_theme = gpd.GeoDataFrame(gdf_loan_theme, geometry='geometry')

# seems like this should work per stack overflow but it puts a lower case nan in and isnull doesn't work properly on it.
#gdf_loan_theme['mpi_region_new'] = np.NaN
gdf_loan_theme = gdf_loan_theme.reindex(columns = np.append(gdf_loan_theme.columns.values, ['mpi_region_new']))

gdf_loan_theme.crs = {"init":epsg}
gdf_loan_theme.head()
### POLYGONS ###
# read in geospatial data
gdf_moz = gpd.GeoDataFrame.from_file("../input/mozambique-geospatial-regions/moz_polbnda_adm2_districts_wfp_ine_pop2012_15_ocha.shp")

# massage regional data
gdf_moz['PROVINCE'] = np.where(gdf_moz['PROVINCE'].str.contains('Zamb'), 'Zambézia', gdf_moz['PROVINCE'])

# aggregate districts into regions used for MPI level in MOZ
moz_regions = {}

provinces = gdf_moz['PROVINCE'].drop_duplicates()
for p in provinces:
    polys = gdf_moz[gdf_moz['PROVINCE'] == p]['geometry']
    u = cascaded_union(polys)
    moz_regions[p] = u
    
#make a geodataframe for the regions    
s = pd.Series(moz_regions, name='geometry')
s.index.name = 'mpi_region_new'
s.reset_index()

gdf_moz = gpd.GeoDataFrame(s, geometry='geometry')
#gdf_moz.crs = {"init":'42106'}
gdf_moz.crs = {"init":epsg}
gdf_moz.reset_index(level=0, inplace=True)

#assign regional MPI to regions
gdf_moz = gdf_moz.merge(df_mpi_subntl[df_mpi_subntl['ISO country code'] == ISO][['Sub-national region', 'MPI Regional']], how='left', 
                                      left_on='mpi_region_new', right_on='Sub-national region')
#manual updates due to character or spelling differences
gdf_moz['MPI Regional'] = np.where(gdf_moz['mpi_region_new'] == 'Zambézia', 0.528, gdf_moz['MPI Regional'])
gdf_moz['MPI Regional'] = np.where(gdf_moz['mpi_region_new'] == 'Maputo', 0.133, gdf_moz['MPI Regional'])
gdf_moz['MPI Regional'] = np.where(gdf_moz['mpi_region_new'] == 'Maputo City', 0.043, gdf_moz['MPI Regional'])
gdf_moz = gdf_moz[['mpi_region_new', 'MPI Regional', 'geometry']]
gdf_moz.head()
### subset points, set in regions
gdf_regions = gdf_moz
gdf_points = gdf_loan_theme[gdf_loan_theme['ISO'] == ISO]

### POINTS IN POLYGONS
for i in range(0, len(gdf_regions)):
    print('i is: ' + str(i) + ' at ' + strftime("%Y-%m-%d %H:%M:%S", gmtime()))
    gdf_points['r_map'] = gdf_points.within(gdf_regions['geometry'][i])
    gdf_points['mpi_region_new'] = np.where(gdf_points['r_map'], gdf_regions['mpi_region_new'][i], gdf_points['mpi_region_new'])
#gdf_points[['id', 'mpi_region_new']].to_csv(ISO + '_pip_output.csv', index = False)
gdf_points.groupby(['mpi_region', 'mpi_region_new'])[['number', 'amount']].sum().reset_index().sort_values('number', ascending=False)
# Merge dataframes
# why did this NaN that worked as null turn into a nan that i'm forced to treat like a string??
LTsubnat = gdf_points[~(gdf_points['mpi_region_new'] == 'nan')]
LTsubnat = LTsubnat.merge(gdf_regions[['mpi_region_new', 'MPI Regional']], on='mpi_region_new')

#~ Get total volume and average MPI Regional Score for each partner loan theme
LS = LTsubnat.groupby(['Partner ID', 'ISO']).agg({'MPI Regional': np.mean, 'amount': np.sum, 'number': np.sum})
#~ Get a volume-weighted average of partners loanthemes.
weighted_avg_LTsubnat = lambda df: np.average(df['MPI Regional'], weights=df['amount'])
#~ and get weighted average for partners. 
MPI_reg_reassign = LS.groupby(level=['Partner ID', 'ISO']).apply(weighted_avg_LTsubnat)
MPI_reg_reassign = MPI_reg_reassign.to_frame()
MPI_reg_reassign.reset_index(level=1, inplace=True)
MPI_reg_reassign = MPI_reg_reassign.rename(index=str, columns={0: 'MPI Score'})

MPI_reg_reassign[MPI_reg_reassign['ISO'] == ISO].reset_index()
Blues = plt.get_cmap('Blues')

regions = gdf_regions['mpi_region_new']

fig, ax = plt.subplots(1, figsize=(12,12))

for r in regions:
    gdf_regions[gdf_regions['mpi_region_new'] == r].plot(ax=ax, color=Blues(gdf_regions[gdf_regions['mpi_region_new'] == r]['MPI Regional']*1.3))

gdf_points[(gdf_points['lat'] != -999) & (gdf_points['Partner ID'] == 23)].plot(ax=ax, markersize=10, color='red', label='23')
gdf_points[(gdf_points['lat'] != -999) & (gdf_points['Partner ID'] == 210)].plot(ax=ax, markersize=10, color='lime', label='210')
gdf_points[(gdf_points['lat'] != -999) & (gdf_points['Partner ID'] == 366)].plot(ax=ax, markersize=10, color='green', label='366')
gdf_points[(gdf_points['lat'] != -999) & (gdf_points['Partner ID'] == 468)].plot(ax=ax, markersize=10, color='yellow', label='468')
gdf_points[(gdf_points['lat'] != -999) & (gdf_points['Partner ID'] == 492)].plot(ax=ax, markersize=10, color='purple', label='492')
#gdf_points[(gdf_points['lat'] != -999) & (gdf_points['Partner ID'] == 261)].plot(ax=ax, markersize=10, color='orange', label='261')


for i, point in gdf_regions.centroid.iteritems():
    reg_n = gdf_regions.iloc[i]['mpi_region_new']
    reg_n = gdf_regions.loc[i, 'mpi_region_new']
    ax.text(s=reg_n, x=point.x, y=point.y, fontsize='large')
    

ax.set_title('Loans across Mozambique by Field Partner\nDarker = Higher MPI.  Range from 0.043 (Maputo City) to 0.528 (Zambézia)')
ax.legend(loc='upper left', frameon=True)
leg = ax.get_legend()
new_title = 'Partner ID'
leg.set_title(new_title)

plt.show()
### POINTS ###
# loan geodataframe
gdf_kv_loans = df_kv_loans.merge(df_kiv_loc, on=['region', 'country'], how='left')
gdf_kv_loans['geometry'] = gdf_kv_loans.apply(lambda row: Point(row['lng'], row['lat']), axis=1)
gdf_kv_loans = gpd.GeoDataFrame(gdf_kv_loans, geometry='geometry')

# seems like this should work per stack overflow but it puts a lower case nan in and isnull doesn't work properly on it.
#gdf_loan_theme['mpi_region_new'] = np.NaN
gdf_kv_loans = gdf_kv_loans.reindex(columns = np.append(gdf_kv_loans.columns.values, ['mpi_region_new']))

gdf_kv_loans.crs = {"init":epsg}
gdf_kv_loans.head()
gdf_points = gdf_kv_loans[gdf_kv_loans['country'] == 'Mozambique']
gdf_points = gdf_points.rename(index=str, columns={'partner_id': 'Partner ID'})

### POINTS IN POLYGONS
for i in range(0, len(gdf_regions)):
    print('i is: ' + str(i) + ' at ' + strftime("%Y-%m-%d %H:%M:%S", gmtime()))
    gdf_points['r_map'] = gdf_points.within(gdf_regions['geometry'][i])
    gdf_points['mpi_region_new'] = np.where(gdf_points['r_map'], gdf_regions['mpi_region_new'][i], gdf_points['mpi_region_new'])
gdf_points[['id', 'mpi_region_new']].to_csv(ISO + '_loans_pip_output.csv', index = False)
# Merge dataframes
# why did this NaN that worked as null turn into a nan that i'm forced to treat like a string??
LTsubnat = gdf_points[~(gdf_points['mpi_region_new'] == 'nan')]
LTsubnat = LTsubnat.merge(gdf_regions[['mpi_region_new', 'MPI Regional']], on='mpi_region_new')

LTsubnat['number'] = 1

#~ Get total volume and average MPI Regional Score for each partner loan theme
LS = LTsubnat.groupby(['Partner ID']).agg({'MPI Regional': np.mean, 'loan_amount': np.sum, 'number': np.sum})
#~ Get a volume-weighted average of partners loanthemes.
weighted_avg_LTsubnat = lambda df: np.average(df['MPI Regional'], weights=df['loan_amount'])
#~ and get weighted average for partners. 
MPI_loan_reassign = LS.groupby(level=['Partner ID']).apply(weighted_avg_LTsubnat)
MPI_loan_reassign = MPI_loan_reassign.to_frame()
MPI_loan_reassign.reset_index(level=0, inplace=True)
MPI_loan_reassign = MPI_loan_reassign.rename(index=str, columns={0: 'MPI Score'})

MPI_loan_reassign
gdf_out = gdf_points[~(gdf_points['mpi_region_new'] == 'nan')].merge(gdf_regions[['mpi_region_new', 'MPI Regional']], on='mpi_region_new')
gdf_out['MPI Weight'] = gdf_out['loan_amount'] * gdf_out['MPI Regional']
gdf_out = gdf_out.groupby('Partner ID')[['loan_amount', 'MPI Weight']].sum().reset_index()
gdf_out['MPI Score'] = gdf_out['MPI Weight'] / gdf_out['loan_amount']
gdf_out[['Partner ID', 'MPI Score']]
Blues = plt.get_cmap('Blues')

regions = gdf_regions['mpi_region_new']

fig, ax = plt.subplots(1, figsize=(12,12))

for r in regions:
    gdf_regions[gdf_regions['mpi_region_new'] == r].plot(ax=ax, color=Blues(gdf_regions[gdf_regions['mpi_region_new'] == r]['MPI Regional']*1.3))

gdf_points[(gdf_points['lat'] != -999) & (gdf_points['Partner ID'] == 23)].plot(ax=ax, markersize=10, color='red', label='23')
gdf_points[(gdf_points['lat'] != -999) & (gdf_points['Partner ID'] == 210)].plot(ax=ax, markersize=10, color='lime', label='210')
gdf_points[(gdf_points['lat'] != -999) & (gdf_points['Partner ID'] == 366)].plot(ax=ax, markersize=10, color='green', label='366')
gdf_points[(gdf_points['lat'] != -999) & (gdf_points['Partner ID'] == 468)].plot(ax=ax, markersize=10, color='yellow', label='468')
gdf_points[(gdf_points['lat'] != -999) & (gdf_points['Partner ID'] == 492)].plot(ax=ax, markersize=10, color='purple', label='492')
gdf_points[(gdf_points['lat'] != -999) & (gdf_points['Partner ID'] == 261)].plot(ax=ax, markersize=10, color='orange', label='261')


for i, point in gdf_regions.centroid.iteritems():
    reg_n = gdf_regions.iloc[i]['mpi_region_new']
    reg_n = gdf_regions.loc[i, 'mpi_region_new']
    ax.text(s=reg_n, x=point.x, y=point.y, fontsize='large')
    

ax.set_title('Loans across Mozambique by Field Partner\nDarker = Higher MPI.  Range from 0.043 (Maputo City) to 0.528 (Zambézia)')
ax.legend(loc='upper left', frameon=True)
leg = ax.get_legend()
new_title = 'Partner ID'
leg.set_title(new_title)

plt.show()
# national scores
df_curr_ntl_mpi = LT[LT['ISO'] == ISO][['Partner ID', 'MPI Score']].drop_duplicates()
df_curr_ntl_mpi['method'] = 'Existing National (rural_percentage)'

# subnational scores
df_curr_subnat = MPI_regional_scores[MPI_regional_scores['ISO'] == ISO].reset_index()
df_curr_subnat = df_curr_subnat[['Partner ID', 'MPI Score']]
df_curr_subnat['method'] = 'Existing Sub-National (Partner Regions)'

# amended region theme scores
df_amend_fld_rgn = MPI_reg_reassign[MPI_reg_reassign['ISO'] == ISO].reset_index()[['Partner ID', 'MPI Score']]
df_amend_fld_rgn['method'] = 'Amended Sub-National (Partner Regions)'

# amended loan scores - averaged
MPI_loan_reassign['method'] = 'Amended Sub-National (Loans - Mean)'

# amended loan scores - weighted
gdf_out = gdf_out[['Partner ID', 'MPI Score']]
gdf_out['method'] = 'Amended Sub-National (Loans - Weighted)'

# combine for comparison
frames = (df_curr_ntl_mpi, df_curr_subnat, df_amend_fld_rgn, MPI_loan_reassign, gdf_out)
df_compare = pd.concat(frames)
df_compare['Partner ID'] = df_compare['Partner ID'].astype(str).str.split('.', 1).str[0]
fig, ax = plt.subplots(figsize=(15, 10))
sns.set_palette('muted')
sns.barplot(x='Partner ID', y='MPI Score', data=df_compare, hue='method')

ax.legend(ncol=1, loc='upper right', frameon=True)
ax.set(ylabel='MPI Score',
       xlabel='Partner ID')

leg = ax.get_legend()
new_title = 'Method'
leg.set_title(new_title)

ax.set_title('Existing vs. Amended Field Partner MPI - Mozambique', fontsize=15)
plt.show()
ISO = 'KEN'
epsg = '4210'

### POLYGONS ###
# read in geospatial data
gdf_ken = gpd.GeoDataFrame.from_file("../input/kenya-geospatial-administrative-regions/ke_district_boundaries.shp")
gdf_ken['DISTNAME'] = gdf_ken['DISTNAME'].str.title()
gdf_ken = gdf_ken.reindex(columns = np.append(gdf_ken.columns.values, ['mpi_region_new']))

districts = ['Mombasa', 'Kwale', 'Kilifi', 'Tana River', 'Lamu', 'Taita Taveta', 'Malindi']
for d in districts:
    gdf_ken['mpi_region_new'] = np.where(gdf_ken['DISTNAME'] == d, 'Coast', gdf_ken['mpi_region_new'])
    
districts = ['Garissa', 'Wajir', 'Mandera']
for d in districts:
    gdf_ken['mpi_region_new'] = np.where(gdf_ken['DISTNAME'] == d, 'North Eastern', gdf_ken['mpi_region_new'])
    
districts = ['Marsabit', 'Isiolo', 'Meru', 'Tharaka-Nithi', 'Embu', 'Kitui', 'Machakos', 'Makueni', 'Meru South', 'Meru North', 'Meru Central', 'Tharaka', 'Mbeere', 'Moyale', 'Mwingi']
for d in districts:
    gdf_ken['mpi_region_new'] = np.where(gdf_ken['DISTNAME'] == d, 'Eastern', gdf_ken['mpi_region_new'])
    
districts = ['Turkana', 'West Pokot', 'Samburu', 'Trans-Nzoia', 'Uasin Gishu', 'Elgeyo-Marakwet', 'Nandi', 'Trans Nzoia', 'Keiyo', 'Koibatek', 'Marakwet', 'Baringo', 'Laikipia', 'Nakuru', 'Narok', 'Kajiado', 'Kericho', 'Bomet', 'Buret', 'Trans Mara']
for d in districts:
    gdf_ken['mpi_region_new'] = np.where(gdf_ken['DISTNAME'] == d, 'Rift Valley', gdf_ken['mpi_region_new'])
    
districts = ['Kakamega', 'Vihiga', 'Bungoma', 'Busia', 'Butere/Mumias', 'Lugari', 'Mt Elgon', 'Teso']
for d in districts:
    gdf_ken['mpi_region_new'] = np.where(gdf_ken['DISTNAME'] == d, 'Western', gdf_ken['mpi_region_new'])
    
districts = ['Siaya', 'Kisumu', 'Homa Bay', 'Migori', 'Kisii', 'Nyamira', 'Bondo', 'Central Kisii', 'Gucha', 'Kuria', 'Nyando', 'Rachuonyo', 'Suba']
for d in districts:
    gdf_ken['mpi_region_new'] = np.where(gdf_ken['DISTNAME'] == d, 'Nyanza', gdf_ken['mpi_region_new'])
    
districts = ['Nairobi']
for d in districts:
    gdf_ken['mpi_region_new'] = np.where(gdf_ken['DISTNAME'] == d, 'Nairobi', gdf_ken['mpi_region_new'])
    
districts = ['Nyandarua', 'Nyeri', 'Kirinyaga', 'Muranga', 'Kiambu', 'Maragua', 'Thika']
for d in districts:
    gdf_ken['mpi_region_new'] = np.where(gdf_ken['DISTNAME'] == d, 'Central', gdf_ken['mpi_region_new'])

# aggregate districts into regions used for MPI level in MOZ
ken_regions = {}
gdf_ken_old_districts = gdf_ken

districts = gdf_ken['mpi_region_new'].drop_duplicates()
for d in districts:
    polys = gdf_ken[gdf_ken['mpi_region_new'] == d]['geometry']
    u = cascaded_union(polys)
    ken_regions[d] = u
    
#make a geodataframe for the regions    
s = pd.Series(ken_regions, name='geometry')
s.index.name = 'mpi_region_new'
s.reset_index()

gdf_ken = gpd.GeoDataFrame(s, geometry='geometry')
gdf_ken.crs = {"init":epsg}
gdf_ken.reset_index(level=0, inplace=True)

#assign regional MPI to regions
gdf_ken = gdf_ken.merge(df_mpi_subntl[df_mpi_subntl['ISO country code'] == ISO][['Sub-national region', 'MPI Regional']], how='left', 
                                      left_on='mpi_region_new', right_on='Sub-national region')

### subset points, set in regions
gdf_regions = gdf_ken
gdf_points = gdf_loan_theme[gdf_loan_theme['ISO'] == ISO]

### POINTS IN POLYGONS
for i in range(0, len(gdf_regions)):
    print('i is: ' + str(i) + ' at ' + strftime("%Y-%m-%d %H:%M:%S", gmtime()))
    gdf_points['r_map'] = gdf_points.within(gdf_regions['geometry'][i])
    gdf_points['mpi_region_new'] = np.where(gdf_points['r_map'], gdf_regions['mpi_region_new'][i], gdf_points['mpi_region_new'])
#gdf_points[['id', 'mpi_region_new']].to_csv(ISO + '_pip_output.csv', index = False)
# amended sub-national partner regions
LTsubnat = gdf_points[~(gdf_points['mpi_region_new'] == 'nan')]
LTsubnat = LTsubnat.merge(gdf_regions[['mpi_region_new', 'MPI Regional']], on='mpi_region_new')

#~ Get total volume and average MPI Regional Score for each partner loan theme
LS = LTsubnat.groupby(['Partner ID', 'ISO']).agg({'MPI Regional': np.mean, 'amount': np.sum, 'number': np.sum})
#~ Get a volume-weighted average of partners loanthemes.
weighted_avg_LTsubnat = lambda df: np.average(df['MPI Regional'], weights=df['amount'])
#~ and get weighted average for partners. 
MPI_reg_reassign = LS.groupby(level=['Partner ID', 'ISO']).apply(weighted_avg_LTsubnat)
MPI_reg_reassign = MPI_reg_reassign.to_frame()
MPI_reg_reassign.reset_index(level=1, inplace=True)
MPI_reg_reassign = MPI_reg_reassign.rename(index=str, columns={0: 'MPI Score'})

MPI_reg_reassign[MPI_reg_reassign['ISO'] == ISO].reset_index()
# now let's do loans
gdf_points = gdf_kv_loans[gdf_kv_loans['country'] == 'Kenya']
gdf_points = gdf_points.rename(index=str, columns={'partner_id': 'Partner ID'})

### POINTS IN POLYGONS
# commented out for speed sake of me working on my own work in this kernel
#for i in range(0, len(gdf_regions)):
#    print('i is: ' + str(i) + ' at ' + strftime("%Y-%m-%d %H:%M:%S", gmtime()))
#    gdf_points['r_map'] = gdf_points.within(gdf_regions['geometry'][i])
#    gdf_points['mpi_region_new'] = np.where(gdf_points['r_map'], gdf_regions['mpi_region_new'][i], gdf_points['mpi_region_new'])
#gdf_points[['id', 'mpi_region_new']].to_csv(ISO + '_loans_pip_output.csv', index = False)

# leverage helper file
gdf_points.drop(columns=['mpi_region_new'], inplace=True)
df_mpi_helper = pd.read_csv("../input/dydkivahelper/KEN_loans_pip_output.csv")
gdf_points = gdf_points.merge(df_mpi_helper, how='left', on='id')
# amended sub-national loans - average
LTsubnat = gdf_points[~(gdf_points['mpi_region_new'] == 'nan') & ~(gdf_points['mpi_region_new'].isnull())]
LTsubnat = LTsubnat.merge(gdf_regions[['mpi_region_new', 'MPI Regional']], on='mpi_region_new')

LTsubnat['number'] = 1

#~ Get total volume and average MPI Regional Score for each partner loan theme
LS = LTsubnat.groupby(['Partner ID']).agg({'MPI Regional': np.mean, 'loan_amount': np.sum, 'number': np.sum})
#~ Get a volume-weighted average of partners loanthemes.
weighted_avg_LTsubnat = lambda df: np.average(df['MPI Regional'], weights=df['loan_amount'])
#~ and get weighted average for partners. 
MPI_loan_reassign = LS.groupby(level=['Partner ID']).apply(weighted_avg_LTsubnat)
MPI_loan_reassign = MPI_loan_reassign.to_frame()
MPI_loan_reassign.reset_index(level=0, inplace=True)
MPI_loan_reassign = MPI_loan_reassign.rename(index=str, columns={0: 'MPI Score'})

MPI_loan_reassign
# amended sub-national loans - weighted
gdf_out = gdf_points[~(gdf_points['mpi_region_new'] == 'nan')].merge(gdf_regions[['mpi_region_new', 'MPI Regional']], on='mpi_region_new')
gdf_out['MPI Weight'] = gdf_out['loan_amount'] * gdf_out['MPI Regional']
gdf_out = gdf_out.groupby('Partner ID')[['loan_amount', 'MPI Weight']].sum().reset_index()
gdf_out['MPI Score'] = gdf_out['MPI Weight'] / gdf_out['loan_amount']
gdf_out[['Partner ID', 'MPI Score']]
Blues = plt.get_cmap('Blues')

regions = gdf_regions['mpi_region_new']

fig, ax = plt.subplots(1, figsize=(12,12))

for r in regions:
    gdf_regions[gdf_regions['mpi_region_new'] == r].plot(ax=ax, color=Blues(gdf_regions[gdf_regions['mpi_region_new'] == r]['MPI Regional']*1.3))

gdf_points[(gdf_points['lng'] > -60) & (gdf_points['lng'] < 44)].plot(ax=ax, markersize=10, color='red', label='loan')
#gdf_points[(gdf_points['lat'] != -999) & (gdf_points['Partner ID'] == 436)].plot(ax=ax, markersize=10, color='lime', label='436')
#gdf_points[(gdf_points['lat'] != -999) & (gdf_points['Partner ID'] == 366)].plot(ax=ax, markersize=10, color='green', label='366')
#gdf_points[(gdf_points['lat'] != -999) & (gdf_points['Partner ID'] == 468)].plot(ax=ax, markersize=10, color='yellow', label='468')
#gdf_points[(gdf_points['lat'] != -999) & (gdf_points['Partner ID'] == 492)].plot(ax=ax, markersize=10, color='purple', label='492')
#gdf_points[(gdf_points['lat'] != -999) & (gdf_points['Partner ID'] == 261)].plot(ax=ax, markersize=10, color='orange', label='261')


for i, point in gdf_regions.centroid.iteritems():
    reg_n = gdf_regions.iloc[i]['mpi_region_new']
    reg_n = gdf_regions.loc[i, 'mpi_region_new']
    ax.text(s=reg_n, x=point.x, y=point.y, fontsize='large')
    

ax.set_title('Loans across Kenya by Field Partner\nDarker = Higher MPI.  Range from 0.020 (Nairobi) to 0.509 (North Eastern)')
#ax.legend(loc='upper left', frameon=True)
#leg = ax.get_legend()
#new_title = 'Partner ID'
#leg.set_title(new_title)

plt.show()
#kenya_only_partners = ['133.0', '138.0', '156.0', '164.0', '218.0', '258.0', '262.0', '276.0', '322.0', '340.0', '386.0', '388.0', '405.0', '436.0', '469.0', '473.0', '491.0', '500.0', '502.0', '505.0', '512.0', '520.0', '526.0', '529.0', '540.0']
#kenya_only_partners = [133, 138, 156, 164, 218, 258, 262, 276, 322, 340, 386, 388, 405, 436, 469, 473, 491, 500, 502, 505, 512, 520, 526, 529, 540]
kenya_only_partners = ['133', '138', '156', '164', '218', '258', '262', '276', '322', '340', '386', '388', '405', '436', '469', '473', '491', '500', '502', '505', '512', '520', '526', '529', '540']

# national scores
df_curr_ntl_mpi = LT[LT['ISO'] == ISO][['Partner ID', 'MPI Score']].drop_duplicates()
df_curr_ntl_mpi['method'] = 'Existing National (rural_percentage)'

# subnational scores
df_curr_subnat = MPI_regional_scores[MPI_regional_scores['ISO'] == ISO].reset_index()
df_curr_subnat = df_curr_subnat[['Partner ID', 'MPI Score']]
df_curr_subnat['method'] = 'Existing Sub-National (Partner Regions)'

# amended region theme scores
df_amend_fld_rgn = MPI_reg_reassign[MPI_reg_reassign['ISO'] == ISO].reset_index()[['Partner ID', 'MPI Score']]
df_amend_fld_rgn['method'] = 'Amended Sub-National (Partner Regions)'

# amended loan scores - averaged
MPI_loan_reassign['method'] = 'Amended Sub-National (Loans - Mean)'

# amended loan scores - weighted
gdf_out = gdf_out[['Partner ID', 'MPI Score']]
gdf_out['method'] = 'Amended Sub-National (Loans - Weighted)'

# combine for comparison
frames = (df_curr_ntl_mpi, df_curr_subnat, df_amend_fld_rgn, MPI_loan_reassign, gdf_out)
df_compare = pd.concat(frames)
df_compare['Partner ID'] = df_compare['Partner ID'].astype(str).str.split('.', 1).str[0]
df_compare = df_compare[df_compare['Partner ID'].isin(kenya_only_partners)]

fig, ax = plt.subplots(figsize=(20, 10))
sns.set_palette('muted')
sns.barplot(x='Partner ID', y='MPI Score', data=df_compare, hue='method')

ax.legend(ncol=1, loc='upper left', frameon=True)
ax.set(ylabel='MPI Score',
       xlabel='Partner ID')

leg = ax.get_legend()
new_title = 'Method'
leg.set_title(new_title)

ax.set_title('Existing vs. Amended Field Partner MPI - Kenya', fontsize=15)
plt.show()
ISO = 'KEN'
epsg = '4210'

### POLYGONS ###
# read in geospatial data
gdf_ken = gpd.GeoDataFrame.from_file("../input/kenya-geospatial-administrative-regions/ke_district_boundaries.shp")
gdf_ken['DISTNAME'] = gdf_ken['DISTNAME'].str.title()
gdf_ken['DISTNAME'] = np.where(gdf_ken['DISTNAME'].str.contains('Taita Taveta'), 'Taita/Taveta', gdf_ken['DISTNAME'])
gdf_ken['DISTNAME'] = np.where(gdf_ken['DISTNAME'].str.contains('Nairobi'), 'Nairobi City', gdf_ken['DISTNAME'])

gdf_ken = gdf_ken.merge(df_pov_ken_ovrl, how='left', left_on='DISTNAME', right_on='residence_county')

# additional manual assignment
districts = ['Butere/Mumias', 'Lugari']
for d in districts:
    gdf_ken['residence_county'] = np.where(gdf_ken['DISTNAME'] == d, 'Kakamega', gdf_ken['residence_county'])
 
districts = ['Central Kisii', 'Gucha']
for d in districts:
    gdf_ken['residence_county'] = np.where(gdf_ken['DISTNAME'] == d, 'Kisii', gdf_ken['residence_county'])    
    
districts = ['Keiyo', 'Marakwet']
for d in districts:
    gdf_ken['residence_county'] = np.where(gdf_ken['DISTNAME'] == d, 'Elgeyo/Marakwet', gdf_ken['residence_county'])   
    
districts = ['Meru Central', 'Meru North']
for d in districts:
    gdf_ken['residence_county'] = np.where(gdf_ken['DISTNAME'] == d, 'Meru', gdf_ken['residence_county'])       

districts = ['Rachuonyo', 'Suba']
for d in districts:
    gdf_ken['residence_county'] = np.where(gdf_ken['DISTNAME'] == d, 'Homa Bay', gdf_ken['residence_county']) 

# Buret - In 2010, the district was split between Kericho County and Bomet County.  just putting this in Kericho to make my life easier
gdf_ken['residence_county'] = np.where(gdf_ken['DISTNAME'].str.contains('Buret'), 'Kericho', gdf_ken['residence_county'])
# The Iteso in Kenya, numbering about 578,000, live mainly in Busia county.
gdf_ken['residence_county'] = np.where(gdf_ken['DISTNAME'].str.contains('Teso'), 'Busia', gdf_ken['residence_county'])
gdf_ken['residence_county'] = np.where(gdf_ken['DISTNAME'].str.contains('Bondo'), 'Siaya', gdf_ken['residence_county'])
gdf_ken['residence_county'] = np.where(gdf_ken['DISTNAME'].str.contains('Koibatek'), 'Baringo', gdf_ken['residence_county'])
gdf_ken['residence_county'] = np.where(gdf_ken['DISTNAME'].str.contains('Kuria'), 'Migori', gdf_ken['residence_county'])
gdf_ken['residence_county'] = np.where(gdf_ken['DISTNAME'].str.contains('Malindi'), 'Kilifi', gdf_ken['residence_county'])
gdf_ken['residence_county'] = np.where(gdf_ken['DISTNAME'].str.contains('Maragua'), 'Muranga', gdf_ken['residence_county'])
gdf_ken['residence_county'] = np.where(gdf_ken['DISTNAME'].str.contains('Mbeere'), 'Embu', gdf_ken['residence_county'])
gdf_ken['residence_county'] = np.where(gdf_ken['DISTNAME'].str.contains('Meru South'), 'Tharaka-Nithi', gdf_ken['residence_county'])
gdf_ken['residence_county'] = np.where(gdf_ken['DISTNAME'].str.contains('Moyale'), 'Marsabit', gdf_ken['residence_county'])
gdf_ken['residence_county'] = np.where(gdf_ken['DISTNAME'].str.contains('Mt Elgon'), 'Bungoma', gdf_ken['residence_county'])
gdf_ken['residence_county'] = np.where(gdf_ken['DISTNAME'].str.contains('Mwingi'), 'Kitui', gdf_ken['residence_county'])
gdf_ken['residence_county'] = np.where(gdf_ken['DISTNAME'].str.contains('Nyando'), 'Kisumu', gdf_ken['residence_county'])
gdf_ken['residence_county'] = np.where(gdf_ken['DISTNAME'].str.contains('Tharaka'), 'Tharaka-Nithi', gdf_ken['residence_county'])
gdf_ken['residence_county'] = np.where(gdf_ken['DISTNAME'].str.contains('Thika'), 'Kiambu', gdf_ken['residence_county'])
gdf_ken['residence_county'] = np.where(gdf_ken['DISTNAME'].str.contains('Trans Mara'), 'Narok', gdf_ken['residence_county'])

# aggregate shapes into counties
ken_regions = {}

districts = gdf_ken['residence_county'].drop_duplicates()
for d in districts:
    polys = gdf_ken[gdf_ken['residence_county'] == d]['geometry']
    u = cascaded_union(polys)
    ken_regions[d] = u
    
# make a geodataframe for the regions    
s = pd.Series(ken_regions, name='geometry')
s.index.name = 'residence_county'
s.reset_index()

gdf_ken = gpd.GeoDataFrame(s, geometry='geometry')
gdf_ken.crs = {"init":epsg}
gdf_ken.reset_index(level=0, inplace=True)
### POINTS IN POLYGONS
# (our points are still kenya)
# set regions to kenya counties
gdf_regions = gdf_ken

gdf_points = gdf_points.reindex(columns = np.append(gdf_points.columns.values, ['residence_county']))

### POINTS IN POLYGONS
# commented out for speed sake of me working on my own work in this kernel
#for i in range(0, len(gdf_regions)):
#    print('i is: ' + str(i) + ' at ' + strftime("%Y-%m-%d %H:%M:%S", gmtime()))
#    gdf_points['r_map'] = gdf_points.within(gdf_regions['geometry'][i])
#    gdf_points['residence_county'] = np.where(gdf_points['r_map'], gdf_regions['residence_county'][i], gdf_points['residence_county'])
#gdf_points[['id', 'residence_county']].to_csv(ISO + '_county_loans_pip_output.csv', index = False)

# leverage helper file
gdf_points.drop(columns=['residence_county'], inplace=True)
df_county_helper = pd.read_csv("../input/dydkivahelper/KEN_county_loans_pip_output.csv")
gdf_points = gdf_points.merge(df_county_helper, how='left', on='id')
# add in kenya overall poverty county data
gdf_regions = gdf_ken
gdf_regions = gdf_regions.merge(df_pov_ken_food, on='residence_county')

Colormap = plt.get_cmap('Oranges')

regions = gdf_regions['residence_county']

fig, ax = plt.subplots(1, figsize=(12,12))

for r in regions:
    gdf_regions[gdf_regions['residence_county'] == r].plot(ax=ax, color=Colormap(gdf_regions[gdf_regions['residence_county'] == r]['Severity of Poverty (%)']/100*2))

for i, point in gdf_regions.centroid.iteritems():
    reg_n = gdf_regions.iloc[i]['residence_county']
    reg_n = gdf_regions.loc[i, 'residence_county']
    ax.text(s=reg_n, x=point.x-0.2, y=point.y, fontsize='small')
    
ax.set_title('Severity of Poverty, Food, Counties Across Kenya')
plt.show()
# add in kenya overall poverty county data
gdf_regions = gdf_ken
gdf_regions = gdf_regions.merge(df_pov_ken_food, on='residence_county')

max_value = gdf_regions['Severity of Poverty (%)'].max()
min_value = gdf_regions['Severity of Poverty (%)'].min()
gdf_regions['sev_pov_nrml'] = (gdf_regions['Severity of Poverty (%)'] - min_value) / (max_value - min_value)

Colormap = plt.get_cmap('Oranges')

regions = gdf_regions['residence_county']

fig, ax = plt.subplots(1, figsize=(12,12))

for r in regions:
    gdf_regions[gdf_regions['residence_county'] == r].plot(ax=ax, color=Colormap(gdf_regions[gdf_regions['residence_county'] == r]['sev_pov_nrml']))

for i, point in gdf_regions.centroid.iteritems():
    reg_n = gdf_regions.iloc[i]['residence_county']
    reg_n = gdf_regions.loc[i, 'residence_county']
    ax.text(s=reg_n, x=point.x-0.2, y=point.y, fontsize='small')
    
ax.set_title('Severity of Poverty, Food, Counties Across Kenya')
plt.show()
# add in kenya overall poverty county data
gdf_regions = gdf_ken
gdf_regions = gdf_regions.merge(df_pov_ken_ovrl, on='residence_county')

max_value = gdf_regions['Severity of Poverty (%)'].max()
min_value = gdf_regions['Severity of Poverty (%)'].min()
gdf_regions['sev_pov_nrml'] = (gdf_regions['Severity of Poverty (%)'] - min_value) / (max_value - min_value)

Colormap = plt.get_cmap('Blues')

regions = gdf_regions['residence_county']

fig, ax = plt.subplots(1, figsize=(12,12))

for r in regions:
    gdf_regions[gdf_regions['residence_county'] == r].plot(ax=ax, color=Colormap(gdf_regions[gdf_regions['residence_county'] == r]['sev_pov_nrml']))

for i, point in gdf_regions.centroid.iteritems():
    reg_n = gdf_regions.iloc[i]['residence_county']
    reg_n = gdf_regions.loc[i, 'residence_county']
    ax.text(s=reg_n, x=point.x-0.2, y=point.y, fontsize='small')
    
ax.set_title('Severity of Poverty, Overall, Counties Across Kenya')
plt.show()
# add in kenya overall poverty county data
gdf_regions = gdf_ken
gdf_regions = gdf_regions.merge(df_pov_ken_hrdc, on='residence_county')

max_value = gdf_regions['Severity of Poverty (%)'].max()
min_value = gdf_regions['Severity of Poverty (%)'].min()
gdf_regions['sev_pov_nrml'] = (gdf_regions['Severity of Poverty (%)'] - min_value) / (max_value - min_value)

Colormap = plt.get_cmap('Purples')

regions = gdf_regions['residence_county']

fig, ax = plt.subplots(1, figsize=(12,12))

for r in regions:
    gdf_regions[gdf_regions['residence_county'] == r].plot(ax=ax, color=Colormap(gdf_regions[gdf_regions['residence_county'] == r]['sev_pov_nrml']))

for i, point in gdf_regions.centroid.iteritems():
    reg_n = gdf_regions.iloc[i]['residence_county']
    reg_n = gdf_regions.loc[i, 'residence_county']
    ax.text(s=reg_n, x=point.x-0.2, y=point.y, fontsize='small')
    

ax.set_title('Severity of Poverty, Extreme, Counties Across Kenya')
plt.show()
# add in kenya overall poverty county data
gdf_regions = gdf_ken
gdf_regions = gdf_regions.merge(df_pov_ken_ovrl, on='residence_county')

max_value = gdf_regions['Severity of Poverty (%)'].max()
min_value = gdf_regions['Severity of Poverty (%)'].min()
gdf_regions['sev_pov_nrml'] = (gdf_regions['Severity of Poverty (%)'] - min_value) / (max_value - min_value)

Colormap = plt.get_cmap('Blues')

regions = gdf_regions['residence_county']

fig, ax = plt.subplots(1, figsize=(12,12))

for r in regions:
    gdf_regions[gdf_regions['residence_county'] == r].plot(ax=ax, color=Colormap(gdf_regions[gdf_regions['residence_county'] == r]['sev_pov_nrml']))

gdf_points[(gdf_points['lng'] > -60) & (gdf_points['lng'] < 44)].plot(ax=ax, markersize=10, color='red', label='loan')
    
for i, point in gdf_regions.centroid.iteritems():
    reg_n = gdf_regions.iloc[i]['residence_county']
    reg_n = gdf_regions.loc[i, 'residence_county']
    ax.text(s=reg_n, x=point.x-0.2, y=point.y, fontsize='small')
    
ax.set_title('Severity of Poverty, Overall, Counties Across Kenya - With Loan Locations')
plt.show()
gdf_sum = gdf_points[(gdf_points['lng'] > -60) & (gdf_points['lng'] < 44)][['residence_county']].apply(pd.value_counts).reset_index()
gdf_sum.rename(index=str, columns={'residence_county': 'loan_count'}, inplace=True)
gdf_sum.rename(index=str, columns={'index': 'residence_county'}, inplace=True)
gdf_ken = gdf_ken.merge(gdf_sum, how='left', on='residence_county')
gdf_ken['loan_count'] = gdf_ken['loan_count'].fillna(0)
gdf_ken.sort_values('loan_count').head(20)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 8), sharex=True)

county_order = gdf_regions.sort_values('Severity of Poverty (%)')['residence_county']

sns.barplot(x='residence_county', y='Severity of Poverty (%)', data=gdf_regions, ax=ax1, color='c', order=county_order)
sns.barplot(x='residence_county', y='loan_count', data=gdf_ken, ax=ax2, color='r', order=county_order)

ax1.set_xlabel('')
ax1.set_title('Kenya Overall Severity of Poverty % vs. Loan Count by County', fontsize=15)

plt.xticks(rotation='vertical')

plt.show()
df_pov_ken_crdt.tail()
df_pov_ken_crdt = df_pov_ken_crdt[~((df_pov_ken_crdt['residence_county'].str.contains('National')) | (df_pov_ken_crdt['residence_county'].str.contains('Rural')) | (df_pov_ken_crdt['residence_county'].str.contains('Urban')))]
df_pov_ken_crdt['prop_deny_credit'] = 100 - df_pov_ken_crdt['Proportion of households that sought and accessed credit (%)']
df_pov_ken_crdt['households_deny_credit'] = df_pov_ken_crdt['prop_deny_credit'] / 100.0 * df_pov_ken_crdt['Number of Households that sought credit (ths)'].astype(float) * 1000
df_pov_ken_crdt.head()
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 8), sharex=True)

county_order = df_pov_ken_crdt.sort_values('prop_deny_credit')['residence_county']

sns.barplot(x='residence_county', y='prop_deny_credit', data=df_pov_ken_crdt, ax=ax1, color='lightgreen', order=county_order)
sns.barplot(x='residence_county', y='loan_count', data=gdf_ken, ax=ax2, color='r', order=county_order)

ax1.set_xlabel('')
ax1.set_title('Propotion of Households Denied Credit vs. Loan Count by County', fontsize=15)

plt.xticks(rotation='vertical')

plt.show()
max_value = df_pov_ken_crdt['prop_deny_credit'].max()
min_value = df_pov_ken_crdt['prop_deny_credit'].min()
df_pov_ken_crdt['prop_deny_credit_nrml'] = (df_pov_ken_crdt['prop_deny_credit'] - min_value) / (max_value - min_value)

df_pov_ken_crdt = df_pov_ken_crdt.merge(gdf_regions[['residence_county', 'sev_pov_nrml']], on='residence_county')
df_pov_ken_crdt['fgt_idx'] = (df_pov_ken_crdt['sev_pov_nrml'] +  df_pov_ken_crdt['prop_deny_credit_nrml']) / 2
df_pov_ken_crdt.head()
gdf_fgt = gdf_points[~gdf_points['residence_county'].isnull()].merge(df_pov_ken_crdt[['residence_county', 'fgt_idx']], on='residence_county')
gdf_fgt['fgt_weight'] = gdf_fgt['loan_amount'] * gdf_fgt['fgt_idx']
gdf_fgt = gdf_fgt.groupby('Partner ID')[['loan_amount', 'fgt_weight']].sum().reset_index()
gdf_fgt['FGT Score'] = gdf_fgt['fgt_weight'] / gdf_fgt['loan_amount']
gdf_fgt['Partner ID'] = gdf_fgt['Partner ID'].astype(str).str.split('.', 1).str[0]

gdf_fgt = gdf_fgt[gdf_fgt['Partner ID'].isin(kenya_only_partners)]
gdf_fgt = gdf_fgt.merge(df_compare[df_compare['method'] == 'Amended Sub-National (Loans - Weighted)'][['Partner ID', 'MPI Score']], on='Partner ID')

gdf_fgt[['Partner ID', 'FGT Score']]
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 8), sharex=True)

county_order = gdf_fgt.sort_values('FGT Score')['Partner ID']

sns.barplot(x='Partner ID', y='MPI Score', data=gdf_fgt, ax=ax1, color='deepskyblue', order=county_order)
sns.barplot(x='Partner ID', y='FGT Score', data=gdf_fgt, ax=ax2, color='mediumorchid', order=county_order)

ax1.set_xlabel('')
ax1.set_title('Kenya MPI Score vs. FGT Score by Partner ID', fontsize=15)

plt.show()
sns.jointplot(x='FGT Score', y='MPI Score', data=gdf_fgt, kind='reg')
gdf_ken_old_districts = gdf_ken_old_districts.merge(df_mpi_subntl[df_mpi_subntl['ISO country code'] == ISO][['Sub-national region', 'MPI Regional']], how='left', 
                                      left_on='mpi_region_new', right_on='Sub-national region')

gdf_pair = gdf_regions[['residence_county', 'geometry', 'Distribution of the Poor (%)', 'Poverty Gap (%)', 'Severity of Poverty (%)', 'sev_pov_nrml']]
gdf_pair = gdf_pair.merge(df_pov_ken_crdt[['residence_county', 'prop_deny_credit', 'prop_deny_credit_nrml', 'fgt_idx']], on='residence_county')

gdf_ken_old_districts['DISTNAME'] = np.where(gdf_ken_old_districts['DISTNAME'] == 'Marakwet', 'Elgeyo/Marakwet', gdf_ken_old_districts['DISTNAME'])
gdf_ken_old_districts['DISTNAME'] = np.where(gdf_ken_old_districts['DISTNAME'] == 'Central Kisii', 'Kisii', gdf_ken_old_districts['DISTNAME'])
gdf_ken_old_districts['DISTNAME'] = np.where(gdf_ken_old_districts['DISTNAME'] == 'Meru Central', 'Meru', gdf_ken_old_districts['DISTNAME'])
gdf_ken_old_districts['DISTNAME'] = np.where(gdf_ken_old_districts['DISTNAME'] == 'Nairobi', 'Nairobi City', gdf_ken_old_districts['DISTNAME'])
gdf_ken_old_districts['DISTNAME'] = np.where(gdf_ken_old_districts['DISTNAME'] == 'Taita Taveta', 'Taita/Taveta', gdf_ken_old_districts['DISTNAME'])
gdf_ken_old_districts['DISTNAME'] = np.where(gdf_ken_old_districts['DISTNAME'] == 'Tharaka', 'Tharaka-Nithi', gdf_ken_old_districts['DISTNAME'])

gdf_ken_old_districts = gdf_ken_old_districts.merge(gdf_pair[['residence_county']], left_on='DISTNAME', right_on='residence_county', how='left')
gdf_pair = gdf_pair.merge(gdf_ken_old_districts[['residence_county', 'mpi_region_new', 'MPI Regional']].drop_duplicates(), how='left', on='residence_county')
g = sns.pairplot(gdf_pair[['sev_pov_nrml', 'prop_deny_credit_nrml', 'fgt_idx', 'MPI Regional']])
g.map_upper(sns.kdeplot)
g.map_diag(plt.hist)
g.map_lower(sns.regplot, scatter_kws={'alpha':0.20}, line_kws={'color': 'red'})
g.map_lower(corrfunc)
#g.map_diag(sns.kdeplot, lw=3)
plt.show()
df_results = gdf_pair[gdf_pair['mpi_region_new'] == 'Rift Valley'][['residence_county', 'Severity of Poverty (%)', 'prop_deny_credit_nrml', 'MPI Regional']].merge(df_pov_ken_ovrl[['residence_county', 'Population (ths)', 'Number of Poor (ths)']], on='residence_county')
df_results['Population (ths)'] = df_results['Population (ths)'].str.replace(',', '').astype(int)
df_results.sort_values('Population (ths)', ascending=False)
ISO = 'PHL'
epsg = '3123'

### POLYGONS ###
# read in geospatial data
gdf_phl = gpd.GeoDataFrame.from_file("../input/philippines-geospatial-administrative-regions/ph-regions-2015.shp")

gdf_phl['mpi_region_new'] = gdf_phl['REGION']
gdf_phl['mpi_region_new'] = np.where(gdf_phl['REGION'].str.contains('ARMM'), 'Armm', gdf_phl['mpi_region_new'])
gdf_phl['mpi_region_new'] = np.where(gdf_phl['REGION'].str.contains('Ilocos'), 'Ilocos Region', gdf_phl['mpi_region_new'])
gdf_phl['mpi_region_new'] = np.where(gdf_phl['REGION'].str.contains('Cagaya'), 'Cagayan Valley', gdf_phl['mpi_region_new'])
gdf_phl['mpi_region_new'] = np.where(gdf_phl['REGION'].str.contains('Central Luz'), 'Central Luzon', gdf_phl['mpi_region_new'])
gdf_phl['mpi_region_new'] = np.where(gdf_phl['REGION'].str.contains('CALAB'), 'Calabarzon', gdf_phl['mpi_region_new'])
gdf_phl['mpi_region_new'] = np.where(gdf_phl['REGION'].str.contains('Bicol'), 'Bicol Region', gdf_phl['mpi_region_new'])
gdf_phl['mpi_region_new'] = np.where(gdf_phl['REGION'].str.contains('Western Vi'), 'Western Visayas', gdf_phl['mpi_region_new'])
gdf_phl['mpi_region_new'] = np.where(gdf_phl['REGION'].str.contains('Central Vi'), 'Central Visayas', gdf_phl['mpi_region_new'])
gdf_phl['mpi_region_new'] = np.where(gdf_phl['REGION'].str.contains('Eastern Vi'), 'Eastern Visayas', gdf_phl['mpi_region_new'])
gdf_phl['mpi_region_new'] = np.where(gdf_phl['REGION'].str.contains('Region IX'), 'Zamboanga Peninsula', gdf_phl['mpi_region_new'])
gdf_phl['mpi_region_new'] = np.where(gdf_phl['REGION'].str.contains('Northern Min'), 'Northern Mindanao', gdf_phl['mpi_region_new'])
gdf_phl['mpi_region_new'] = np.where(gdf_phl['REGION'].str.contains('Davao'), 'Davao Peninsula', gdf_phl['mpi_region_new'])
gdf_phl['mpi_region_new'] = np.where(gdf_phl['REGION'].str.contains('SOCC'), 'Soccsksargen', gdf_phl['mpi_region_new'])
gdf_phl['mpi_region_new'] = np.where(gdf_phl['REGION'].str.contains('Caraga'), 'CARAGA', gdf_phl['mpi_region_new'])
gdf_phl['mpi_region_new'] = np.where(gdf_phl['REGION'].str.contains('Cordill'), 'Cordillera Admin Region', gdf_phl['mpi_region_new'])
gdf_phl['mpi_region_new'] = np.where(gdf_phl['REGION'].str.contains('NCR'), 'National Capital Region', gdf_phl['mpi_region_new'])
gdf_phl['mpi_region_new'] = np.where(gdf_phl['REGION'].str.contains('MIMAROPA'), 'Mimaropa', gdf_phl['mpi_region_new'])

gdf_phl = gdf_phl.merge(df_mpi_subntl[df_mpi_subntl['Country'] == 'Philippines'][['Sub-national region', 'MPI Regional']], how='left', left_on='mpi_region_new', right_on='Sub-national region')
#negros island region was made by splitting up visayas, which has the same MPI, which we'll set manually here
gdf_phl['MPI Regional'] = np.where(gdf_phl['REGION'].str.contains('NIR'), 0.055, gdf_phl['MPI Regional'])
gdf_phl.drop(columns=['Sub-national region'], inplace=True)
############## loan theme region partner level
### subset points, set in regions
gdf_regions = gdf_phl
gdf_points = gdf_loan_theme[gdf_loan_theme['ISO'] == ISO]

### POINTS IN POLYGONS
for i in range(0, len(gdf_regions)):
    print('i is: ' + str(i) + ' at ' + strftime("%Y-%m-%d %H:%M:%S", gmtime()))
    gdf_points['r_map'] = gdf_points.within(gdf_regions['geometry'][i])
    gdf_points['mpi_region_new'] = np.where(gdf_points['r_map'], gdf_regions['mpi_region_new'][i], gdf_points['mpi_region_new'])
#gdf_points[['id', 'mpi_region_new']].to_csv(ISO + '_pip_output.csv', index = False)

# amended sub-national partner regions
LTsubnat = gdf_points[~(gdf_points['mpi_region_new'] == 'nan')]
LTsubnat = LTsubnat.merge(gdf_regions[['mpi_region_new', 'MPI Regional']], on='mpi_region_new')

#~ Get total volume and average MPI Regional Score for each partner loan theme
LS = LTsubnat.groupby(['Partner ID', 'ISO']).agg({'MPI Regional': np.mean, 'amount': np.sum, 'number': np.sum})
#~ Get a volume-weighted average of partners loanthemes.
weighted_avg_LTsubnat = lambda df: np.average(df['MPI Regional'], weights=df['amount'])
#~ and get weighted average for partners. 
MPI_reg_reassign = LS.groupby(level=['Partner ID', 'ISO']).apply(weighted_avg_LTsubnat)
MPI_reg_reassign = MPI_reg_reassign.to_frame()
MPI_reg_reassign.reset_index(level=1, inplace=True)
MPI_reg_reassign = MPI_reg_reassign.rename(index=str, columns={0: 'MPI Score'})

MPI_reg_reassign[MPI_reg_reassign['ISO'] == ISO].reset_index()
# now let's do loans
gdf_points = gdf_kv_loans[gdf_kv_loans['country'] == 'Philippines']
gdf_points = gdf_points.rename(index=str, columns={'partner_id': 'Partner ID'})

### POINTS IN POLYGONS
# commented out for speed sake of me working on my own work in this kernel
#for i in range(0, len(gdf_regions)):
#    print('i is: ' + str(i) + ' at ' + strftime("%Y-%m-%d %H:%M:%S", gmtime()))
#    gdf_points['r_map'] = gdf_points.within(gdf_regions['geometry'][i])
#    gdf_points['mpi_region_new'] = np.where(gdf_points['r_map'], gdf_regions['mpi_region_new'][i], gdf_points['mpi_region_new'])
#gdf_points[['id', 'mpi_region_new']].to_csv(ISO + '_loans_pip_output.csv', index = False)

# leverage helper file; get results from first kernel
gdf_points.drop(columns=['mpi_region_new'], inplace=True)
df_mpi_helper = pd.read_csv("../input/philippines-geospatial-administrative-regions/phil_pip_output.csv")
gdf_points = gdf_points.merge(df_mpi_helper, how='left', on='id')
gdf_points = gdf_points.rename(index=str, columns={'region_mpi_new': 'mpi_region_new'})
gdf_points[(gdf_points['lat'] != -999) & ((gdf_points['id'] == 654958) | (gdf_points['id'] == 1326525))][['id', 'region', 'geometry', 'mpi_region_new']]
fig, ax = plt.subplots(1, figsize=(12,12))

gdf_regions[gdf_regions['mpi_region_new'] == 'Central Visayas'].plot(ax=ax, color=Blues(gdf_regions[gdf_regions['mpi_region_new'] == 'Central Visayas']['MPI Regional']*6), label=r)

gdf_points[(gdf_points['lat'] != -999) & ((gdf_points['id'] == 654958) | (gdf_points['id'] == 1326525))].plot(ax=ax, markersize=5, color='red')

plt.show()
### wiggle round 1
gdf_missing = gdf_points[(gdf_points['lat'] != -999) & (gdf_points['mpi_region_new'].isnull())][['id', 'lat', 'lng', 'geometry', 'mpi_region_new']]

wiggle = [(0.002, 0.002), (0.002, -0.002), (-0.002, 0.002), (-0.002, -0.002)]

for w in wiggle:
    gdf_missing['geometry'] = gdf_missing.apply(lambda row: Point(row['lng']+w[0], row['lat']+w[1]), axis=1)
    
    for i in range(0, len(gdf_regions)):
        print('i is: ' + str(i) + ' at ' + strftime("%Y-%m-%d %H:%M:%S", gmtime()))
        #UNCOMMENT BELOW TO RUN CODE
        #gdf_missing['r_map'] = gdf_missing.within(gdf_regions['geometry'][i])
        #gdf_missing['mpi_region_new'] = np.where(((gdf_missing['r_map']) & ((gdf_missing['mpi_region_new'].isnull()) | (gdf_missing['mpi_region_new'] == 'nan'))), gdf_regions['mpi_region_new'][i], gdf_missing['mpi_region_new'])
        
gdf_missing['mpi_region_new_wiggled'] = gdf_missing['mpi_region_new']

# swap comment on below lines to run code or read helper
#gdf_missing[['id', 'mpi_region_new_wiggled']].to_csv(ISO + '_wiggle1_pip_output.csv', index = False)
gdf_missing = pd.read_csv("../input/dydkivahelper/PHL_wiggle1_pip_output.csv")

#merge results in
gdf_points = gdf_points.merge(gdf_missing[~gdf_missing['mpi_region_new_wiggled'].isnull()][['id', 'mpi_region_new_wiggled']], how='left', on='id')
gdf_points['mpi_region_new'] = np.where(~gdf_points['mpi_region_new_wiggled'].isnull(), gdf_points['mpi_region_new_wiggled'], gdf_points['mpi_region_new'])
gdf_points.drop(columns=['mpi_region_new_wiggled'], inplace=True)

### wiggle round 2
gdf_missing = gdf_points[(gdf_points['lat'] != -999) & (gdf_points['mpi_region_new'].isnull())][['id', 'lat', 'lng', 'geometry', 'mpi_region_new']]

wiggle = [(0.01, 0.01), (0.01, -0.01), (-0.01, 0.01), (-0.01, -0.01)]

for w in wiggle:
    gdf_missing['geometry'] = gdf_missing.apply(lambda row: Point(row['lng']+w[0], row['lat']+w[1]), axis=1)
    
    for i in range(0, len(gdf_regions)):
        print('i is: ' + str(i) + ' at ' + strftime("%Y-%m-%d %H:%M:%S", gmtime()))
        #UNCOMMENT BELOW TO RUN CODE
        #gdf_missing['r_map'] = gdf_missing.within(gdf_regions['geometry'][i])
        #gdf_missing['mpi_region_new'] = np.where(((gdf_missing['r_map']) & ((gdf_missing['mpi_region_new'].isnull()) | (gdf_missing['mpi_region_new'] == 'nan'))), gdf_regions['mpi_region_new'][i], gdf_missing['mpi_region_new'])
        
gdf_missing['mpi_region_new_wiggled'] = gdf_missing['mpi_region_new']

# swap comment on below lines to run code or read helper
#gdf_missing[['id', 'mpi_region_new_wiggled']].to_csv(ISO + '_wiggle2_pip_output.csv', index = False)
gdf_missing = pd.read_csv("../input/dydkivahelper/PHL_wiggle2_pip_output.csv")

#merge in round 2
gdf_points = gdf_points.merge(gdf_missing[~gdf_missing['mpi_region_new_wiggled'].isnull()][['id', 'mpi_region_new_wiggled']], how='left', on='id')
gdf_points['mpi_region_new'] = np.where(~gdf_points['mpi_region_new_wiggled'].isnull(), gdf_points['mpi_region_new_wiggled'], gdf_points['mpi_region_new'])
gdf_points.drop(columns=['mpi_region_new_wiggled'], inplace=True)
# amended sub-national loans - average
LTsubnat = gdf_points[~(gdf_points['mpi_region_new'] == 'nan') & ~(gdf_points['mpi_region_new'].isnull())]
LTsubnat = LTsubnat.merge(gdf_regions[['mpi_region_new', 'MPI Regional']], on='mpi_region_new')

LTsubnat['number'] = 1

#~ Get total volume and average MPI Regional Score for each partner loan theme
LS = LTsubnat.groupby(['Partner ID']).agg({'MPI Regional': np.mean, 'loan_amount': np.sum, 'number': np.sum})
#~ Get a volume-weighted average of partners loanthemes.
weighted_avg_LTsubnat = lambda df: np.average(df['MPI Regional'], weights=df['loan_amount'])
#~ and get weighted average for partners. 
MPI_loan_reassign = LS.groupby(level=['Partner ID']).apply(weighted_avg_LTsubnat)
MPI_loan_reassign = MPI_loan_reassign.to_frame()
MPI_loan_reassign.reset_index(level=0, inplace=True)
MPI_loan_reassign = MPI_loan_reassign.rename(index=str, columns={0: 'MPI Score'})

MPI_loan_reassign
# amended sub-national loans - weighted
gdf_out = gdf_points[~(gdf_points['mpi_region_new'] == 'nan')].merge(gdf_regions[['mpi_region_new', 'MPI Regional']], on='mpi_region_new')
gdf_out['MPI Weight'] = gdf_out['loan_amount'] * gdf_out['MPI Regional']
gdf_out = gdf_out.groupby('Partner ID')[['loan_amount', 'MPI Weight']].sum().reset_index()
gdf_out['MPI Score'] = gdf_out['MPI Weight'] / gdf_out['loan_amount']
gdf_out[['Partner ID', 'MPI Score']]
# national scores
df_curr_ntl_mpi = LT[LT['ISO'] == ISO][['Partner ID', 'MPI Score']].drop_duplicates()
df_curr_ntl_mpi['method'] = 'Existing National (rural_percentage)'

# subnational scores
df_curr_subnat = MPI_regional_scores[MPI_regional_scores['ISO'] == ISO].reset_index()
df_curr_subnat = df_curr_subnat[['Partner ID', 'MPI Score']]
df_curr_subnat['method'] = 'Existing Sub-National (Partner Regions)'

# amended region theme scores
df_amend_fld_rgn = MPI_reg_reassign[MPI_reg_reassign['ISO'] == ISO].reset_index()[['Partner ID', 'MPI Score']]
df_amend_fld_rgn['method'] = 'Amended Sub-National (Partner Regions)'

# amended loan scores - averaged
MPI_loan_reassign['method'] = 'Amended Sub-National (Loans - Mean)'

# amended loan scores - weighted
gdf_out = gdf_out[['Partner ID', 'MPI Score']]
gdf_out['method'] = 'Amended Sub-National (Loans - Weighted)'

# combine for comparison
frames = (df_curr_ntl_mpi, df_curr_subnat, df_amend_fld_rgn, MPI_loan_reassign, gdf_out)
df_compare = pd.concat(frames)
df_compare['Partner ID'] = df_compare['Partner ID'].astype(str).str.split('.', 1).str[0]
Blues = plt.get_cmap('Blues')

regions = gdf_regions['mpi_region_new']

fig, ax = plt.subplots(1, figsize=(12,12))

for r in regions:
    gdf_regions[gdf_regions['mpi_region_new'] == r].plot(ax=ax, color=Blues(gdf_regions[gdf_regions['mpi_region_new'] == r]['MPI Regional']*6))

gdf_points[(gdf_points['lng'] != -999)].plot(ax=ax, markersize=10, color='red', alpha=0.2, label='loan')

ax.text(s='Ilocos Region', x=119, y=16.65, fontsize='large')   
ax.text(s='CAR', x=120.85, y=17.5, fontsize='large') 
ax.text(s='Cagayan Valley', x=122.6, y=17, fontsize='large') 
ax.text(s='Central Luzon', x=118.5, y=15, fontsize='large') 
ax.text(s='NCR', x=120.8, y=14.5, fontsize='large') 
ax.text(s='Calabarzon', x=121.4, y=14.4, fontsize='large') 
ax.text(s='Mimaropa', x=119.7, y=12.7, fontsize='large') 
ax.text(s='Bicol Region', x=124.5, y=13.75, fontsize='large') 
ax.text(s='Western Visayas', x=120.4, y=11, fontsize='large') 
ax.text(s='Central Visayas', x=123.5, y=10.4, fontsize='large') 
ax.text(s='Eastern Visayas', x=125.6, y=11.5, fontsize='large') 
ax.text(s='Negros Island Region', x=120.3, y=9.8, fontsize='large') 
ax.text(s='Northern Mindanao', x=123.3, y=8.75, fontsize='large') 
ax.text(s='Davao Peninsula', x=125.6, y=6, fontsize='large') 
ax.text(s='Soccsksargen', x=123.8, y=5.3, fontsize='large') 
ax.text(s='CARAGA', x=126.3, y=9, fontsize='large') 
ax.text(s='Armm', x=123.2, y=6.8, fontsize='large') 
ax.text(s='Zamboanga Peninsula', x=119.8, y=7.5, fontsize='large') 

ax.set_title('Kiva Loans in Philippines Administrative Regions by MPI\nDarker = Higher MPI.  Range from 0.026 (NCR) to 0.140 (ARMM)')

plt.show()
Blues = plt.get_cmap('Blues')

regions = gdf_regions['mpi_region_new']

fig, ax = plt.subplots(1, figsize=(12,12))

for r in regions:
    gdf_regions[gdf_regions['mpi_region_new'] == r].plot(ax=ax, color=Blues(gdf_regions[gdf_regions['mpi_region_new'] == r]['MPI Regional']*6))

gdf_points[(gdf_points['lng'] != -999) & (gdf_points['Partner ID'] == 123)].plot(ax=ax, markersize=10, color='red', label=123)
gdf_points[(gdf_points['lng'] != -999) & (gdf_points['Partner ID'] == 125)].plot(ax=ax, markersize=10, color='yellow', label=125)
gdf_points[(gdf_points['lng'] != -999) & (gdf_points['Partner ID'] == 126)].plot(ax=ax, markersize=10, color='green', label=126)
gdf_points[(gdf_points['lng'] != -999) & (gdf_points['Partner ID'] == 136)].plot(ax=ax, markersize=10, color='orange', label=136)
gdf_points[(gdf_points['lng'] != -999) & (gdf_points['Partner ID'] == 144)].plot(ax=ax, markersize=10, color='purple', label=144)
gdf_points[(gdf_points['lng'] != -999) & (gdf_points['Partner ID'] == 145)].plot(ax=ax, markersize=10, color='pink', label=145)
gdf_points[(gdf_points['lng'] != -999) & (gdf_points['Partner ID'] == 351)].plot(ax=ax, markersize=10, color='lime', label=351)
gdf_points[(gdf_points['lng'] != -999) & (gdf_points['Partner ID'] == 389)].plot(ax=ax, markersize=10, color='darkviolet', label=389)
gdf_points[(gdf_points['lng'] != -999) & (gdf_points['Partner ID'] == 409)].plot(ax=ax, markersize=10, color='crimson', label=409)
gdf_points[(gdf_points['lng'] != -999) & (gdf_points['Partner ID'] == 508)].plot(ax=ax, markersize=10, color='blue', label=508)

ax.text(s='Ilocos Region', x=119, y=16.65, fontsize='large')   
ax.text(s='CAR', x=120.85, y=17.5, fontsize='large') 
ax.text(s='Cagayan Valley', x=122.6, y=17, fontsize='large') 
ax.text(s='Central Luzon', x=118.5, y=15, fontsize='large') 
ax.text(s='NCR', x=120.8, y=14.5, fontsize='large') 
ax.text(s='Calabarzon', x=121.4, y=14.4, fontsize='large') 
ax.text(s='Mimaropa', x=119.7, y=12.7, fontsize='large') 
ax.text(s='Bicol Region', x=124.5, y=13.75, fontsize='large') 
ax.text(s='Western Visayas', x=120.4, y=11, fontsize='large') 
ax.text(s='Central Visayas', x=123.5, y=10.4, fontsize='large') 
ax.text(s='Eastern Visayas', x=125.6, y=11.5, fontsize='large') 
ax.text(s='Negros Island Region', x=120.3, y=9.8, fontsize='large') 
ax.text(s='Northern Mindanao', x=123.3, y=8.75, fontsize='large') 
ax.text(s='Davao Peninsula', x=125.6, y=6, fontsize='large') 
ax.text(s='Soccsksargen', x=123.8, y=5.3, fontsize='large') 
ax.text(s='CARAGA', x=126.3, y=9, fontsize='large') 
ax.text(s='Armm', x=123.2, y=6.8, fontsize='large') 
ax.text(s='Zamboanga Peninsula', x=119.8, y=7.5, fontsize='large') 

ax.set_title('Kiva Loans in Philippines Administrative Regions by MPI\nDarker = Higher MPI.  Range from 0.026 (NCR) to 0.140 (ARMM)')

ax.legend(ncol=2, loc='best', frameon=True)
leg = ax.get_legend()
new_title = 'Partner ID'
leg.set_title(new_title)

plt.show()
fig, ax = plt.subplots(figsize=(20, 10))
sns.set_palette('muted')
sns.barplot(x='Partner ID', y='MPI Score', data=df_compare, hue='method')

ax.legend(ncol=1, loc='upper left', frameon=True)
ax.set(ylabel='MPI Score',
       xlabel='Partner ID')

leg = ax.get_legend()
new_title = 'Method'
leg.set_title(new_title)

ax.set_title('Existing vs. Amended Field Partner MPI - Philippines', fontsize=15)
plt.show()
df_demo_phl
df_demo_phl['region_mpi'] = np.where(df_demo_phl['region'] == 'NEGROS ISLAND REGION', 'Negros Island Region (NIR)', df_demo_phl['region_mpi'])
df_demo_phl['consumption (2016 est 2000 php)'] = np.where(df_demo_phl['region'] == 'NEGROS ISLAND REGION', (48800.0+44685.0)/2, df_demo_phl['consumption (2016 est 2000 php)'])
df_demo_phl['2015 gpd in usd per capita'] = np.where(df_demo_phl['region'] == 'NEGROS ISLAND REGION', (2697.96+3153.59)/2, df_demo_phl['2015 gpd in usd per capita'])
gdf_regions = gdf_regions.merge(df_demo_phl[['region_mpi', '2015 gpd in usd per capita', 'consumption (2016 est 2000 php)', '2015 population']], left_on='mpi_region_new', right_on='region_mpi')
g = sns.pairplot(gdf_regions[['MPI Regional', '2015 gpd in usd per capita', 'consumption (2016 est 2000 php)']])
g.map_upper(sns.kdeplot)
g.map_diag(plt.hist)
g.map_lower(sns.regplot, scatter_kws={'alpha':0.20}, line_kws={'color': 'red'})
g.map_lower(corrfunc)
#g.map_diag(sns.kdeplot, lw=3)
plt.show()
g = sns.pairplot(gdf_regions[gdf_regions['mpi_region_new'] != 'National Capital Region'][['MPI Regional', '2015 gpd in usd per capita', 'consumption (2016 est 2000 php)']])
g.map_upper(sns.kdeplot)
g.map_diag(plt.hist)
g.map_lower(sns.regplot, scatter_kws={'alpha':0.20}, line_kws={'color': 'red'})
g.map_lower(corrfunc)
#g.map_diag(sns.kdeplot, lw=3)
plt.show()
max_value = gdf_regions['consumption (2016 est 2000 php)'].max()
min_value = gdf_regions['consumption (2016 est 2000 php)'].min()
gdf_regions['cons_nrml'] = (gdf_regions['consumption (2016 est 2000 php)'] - min_value) / (max_value - min_value)
gdf_regions['cons_nrml'] = 1 - gdf_regions['cons_nrml']
Blues = plt.get_cmap('Blues')

regions = gdf_regions['mpi_region_new']

fig, ax = plt.subplots(1, figsize=(12,12))

for r in regions:
    gdf_regions[gdf_regions['mpi_region_new'] == r].plot(ax=ax, color=Blues(gdf_regions[gdf_regions['mpi_region_new'] == r]['cons_nrml']))

gdf_points[(gdf_points['lng'] != -999) & (gdf_points['Partner ID'] == 123)].plot(ax=ax, markersize=10, color='red', label=123)
gdf_points[(gdf_points['lng'] != -999) & (gdf_points['Partner ID'] == 125)].plot(ax=ax, markersize=10, color='yellow', label=125)
gdf_points[(gdf_points['lng'] != -999) & (gdf_points['Partner ID'] == 126)].plot(ax=ax, markersize=10, color='green', label=126)
gdf_points[(gdf_points['lng'] != -999) & (gdf_points['Partner ID'] == 136)].plot(ax=ax, markersize=10, color='orange', label=136)
gdf_points[(gdf_points['lng'] != -999) & (gdf_points['Partner ID'] == 144)].plot(ax=ax, markersize=10, color='purple', label=144)
gdf_points[(gdf_points['lng'] != -999) & (gdf_points['Partner ID'] == 145)].plot(ax=ax, markersize=10, color='pink', label=145)
gdf_points[(gdf_points['lng'] != -999) & (gdf_points['Partner ID'] == 351)].plot(ax=ax, markersize=10, color='lime', label=351)
gdf_points[(gdf_points['lng'] != -999) & (gdf_points['Partner ID'] == 389)].plot(ax=ax, markersize=10, color='darkviolet', label=389)
gdf_points[(gdf_points['lng'] != -999) & (gdf_points['Partner ID'] == 409)].plot(ax=ax, markersize=10, color='crimson', label=409)
gdf_points[(gdf_points['lng'] != -999) & (gdf_points['Partner ID'] == 508)].plot(ax=ax, markersize=10, color='blue', label=508)

ax.text(s='Ilocos Region', x=119, y=16.65, fontsize='large')   
ax.text(s='CAR', x=120.85, y=17.5, fontsize='large') 
ax.text(s='Cagayan Valley', x=122.6, y=17, fontsize='large') 
ax.text(s='Central Luzon', x=118.5, y=15, fontsize='large') 
ax.text(s='NCR', x=120.8, y=14.5, fontsize='large') 
ax.text(s='Calabarzon', x=121.4, y=14.4, fontsize='large') 
ax.text(s='Mimaropa', x=119.7, y=12.7, fontsize='large') 
ax.text(s='Bicol Region', x=124.5, y=13.75, fontsize='large') 
ax.text(s='Western Visayas', x=120.4, y=11, fontsize='large') 
ax.text(s='Central Visayas', x=123.5, y=10.4, fontsize='large') 
ax.text(s='Eastern Visayas', x=125.6, y=11.5, fontsize='large') 
ax.text(s='Negros Island Region', x=120.3, y=9.8, fontsize='large') 
ax.text(s='Northern Mindanao', x=123.3, y=8.75, fontsize='large') 
ax.text(s='Davao Peninsula', x=125.6, y=6, fontsize='large') 
ax.text(s='Soccsksargen', x=123.8, y=5.3, fontsize='large') 
ax.text(s='CARAGA', x=126.3, y=9, fontsize='large') 
ax.text(s='Armm', x=123.2, y=6.8, fontsize='large') 
ax.text(s='Zamboanga Peninsula', x=119.8, y=7.5, fontsize='large') 

ax.set_title('Kiva Loans in Philippines Administrative Regions by MPI\nDarker = Lower Consumption Per Capita')

ax.legend(ncol=2, loc='best', frameon=True)
leg = ax.get_legend()
new_title = 'Partner ID'
leg.set_title(new_title)

plt.show()
gdf_cons = gdf_points[~gdf_points['mpi_region_new'].isnull()].merge(gdf_regions[['mpi_region_new', 'cons_nrml']], on='mpi_region_new')
gdf_cons['cons_weight'] = gdf_cons['loan_amount'] * gdf_cons['cons_nrml']
gdf_cons = gdf_cons.groupby('Partner ID')[['loan_amount', 'cons_weight']].sum().reset_index()
gdf_cons['Consumption Score'] = gdf_cons['cons_weight'] / gdf_cons['loan_amount']
gdf_cons['Partner ID'] = gdf_cons['Partner ID'].astype(str).str.split('.', 1).str[0]
gdf_cons = gdf_cons.merge(df_compare[df_compare['method'] == 'Amended Sub-National (Loans - Weighted)'], on='Partner ID')

gdf_sum = gdf_points[['mpi_region_new']].apply(pd.value_counts).reset_index()
gdf_sum.rename(index=str, columns={'mpi_region_new': 'loan_count'}, inplace=True)
gdf_sum.rename(index=str, columns={'index': 'mpi_region_new'}, inplace=True)
gdf_regions = gdf_regions.merge(gdf_sum, how='left', on='mpi_region_new')
gdf_regions['loan_count'] = gdf_regions['loan_count'].fillna(0)
gdf_regions['loans_per_cap'] = gdf_regions['loan_count'] / gdf_regions['2015 population']
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(20, 8), sharex=True)

region_order = gdf_regions.sort_values('cons_nrml')['mpi_region_new']

sns.barplot(x='mpi_region_new', y='cons_nrml', data=gdf_regions, ax=ax1, color='c', order=region_order)
sns.barplot(x='mpi_region_new', y='loan_count', data=gdf_regions, ax=ax2, color='r', order=region_order)
sns.barplot(x='mpi_region_new', y='loans_per_cap', data=gdf_regions, ax=ax3, color='salmon', order=region_order)

ax1.set_xlabel('')
ax1.set_title('Philippines Consumption Per Capita (Normalized and Inverted) vs. Loan Count vs. Loan Count Per Capita by Region', fontsize=15)

plt.xticks(rotation='vertical')

plt.show()
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 8), sharex=True)

region_order = gdf_cons.sort_values('Consumption Score')['Partner ID']

sns.barplot(x='Partner ID', y='MPI Score', data=gdf_cons, ax=ax1, color='deepskyblue', order=region_order)
sns.barplot(x='Partner ID', y='Consumption Score', data=gdf_cons, ax=ax2, color='mediumorchid', order=region_order)

ax1.set_xlabel('')
ax1.set_title('Philippines MPI Score vs. Consumption Score by Partner ID', fontsize=15)

plt.show()
ISO = 'IND'
epsg = '1121'

### POLYGONS ###
# read in geospatial data
gdf_ind = gpd.GeoDataFrame.from_file("../input/india-geospatial-regions/Admin2.shp")
gdf_ind['ST_NM'] = np.where(gdf_ind['ST_NM'] == 'Andaman & Nicobar Island', 'A & N Islands', gdf_ind['ST_NM'])
gdf_ind['ST_NM'] = np.where(gdf_ind['ST_NM'] == 'NCT of Delhi', 'Delhi', gdf_ind['ST_NM'])
gdf_ind['ST_NM'] = np.where(gdf_ind['ST_NM'] == 'Arunanchal Pradesh', 'Arunachal Pradesh', gdf_ind['ST_NM'])
gdf_ind['ST_NM'] = np.where(gdf_ind['ST_NM'] == 'Dadara & Nagar Havelli', 'Dadra & Nagar Haveli', gdf_ind['ST_NM'])
gdf_ind['ST_NM'] = np.where(gdf_ind['ST_NM'] == 'Telangana', 'Andhra Pradesh', gdf_ind['ST_NM'])

# aggregate polygons
ind_regions = {}

districts = gdf_ind['ST_NM'].drop_duplicates()
for d in districts:
    polys = gdf_ind[gdf_ind['ST_NM'] == d]['geometry']
    u = cascaded_union(polys)
    ind_regions[d] = u
    
#make a geodataframe for the regions    
s = pd.Series(ind_regions, name='geometry')
s.index.name = 'ST_NM'
s.reset_index()

gdf_ind = gpd.GeoDataFrame(s, geometry='geometry')
gdf_ind.crs = {"init":epsg}
gdf_ind.reset_index(level=0, inplace=True)
gdf_ind = gdf_ind.rename(index=str, columns={'ST_NM': 'State'})

#rename for join
df_ind_pov['State'] = np.where(df_ind_pov['State'] == 'Orissa/Odisha', 'Odisha', df_ind_pov['State'])
df_ind_pov['State'] = np.where(df_ind_pov['State'] == 'Uttarkhand', 'Uttarakhand', df_ind_pov['State'])

gdf_ind = gdf_ind.merge(df_ind_pov, how='inner', on='State')
gdf_regions = gdf_ind
gdf_points = gdf_kv_loans[gdf_kv_loans['country'] == 'India']
gdf_points = gdf_points.rename(index=str, columns={'partner_id': 'Partner ID'})
gdf_points = gdf_points.reindex(columns = np.append(gdf_points.columns.values, ['State']))

### POINTS IN POLYGONS
# commented out for speed sake of me working on my own work in this kernel
#for i in range(0, len(gdf_regions)):
#    print('i is: ' + str(i) + ' at ' + strftime("%Y-%m-%d %H:%M:%S", gmtime()))
#    gdf_points['r_map'] = gdf_points.within(gdf_regions['geometry'][i])
#    gdf_points['State'] = np.where(gdf_points['r_map'], gdf_regions['State'][i], gdf_points['State'])
#gdf_points[['id', 'State']].to_csv(ISO + '_loans_pip_output.csv', index = False)

# leverage helper file
gdf_points.drop(columns=['State'], inplace=True)
df_ind_helper = pd.read_csv("../input/dydkivahelper/IND_loans_pip_output.csv")
gdf_points = gdf_points.merge(df_ind_helper, how='left', on='id')
gdf_points.head()
regions = gdf_regions['State'] #.sort_values('Headcount Ratio (%)')['State']
palette = itertools.cycle(sns.color_palette('Set3', len(regions)))
#Headcount Ratio (%)

category20 = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c', '#98df8a', '#e7969c', '#ff9896', '#9467bd', '#c5b0d5', '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f',
             '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5', '#9c9ede', '#b5cf6b', '#de9ed6', '#bcbddc', '#c7e9c0', '#d6616b']
palette = itertools.cycle(category20)
#sns.set_palette(flatui)
#sns.lmplot( x="sepal_length", y="sepal_width", data=df, fit_reg=False, hue='species', legend=False)


fig, ax = plt.subplots(1, figsize=(12,12))

for r in regions:
    gdf_regions[gdf_regions['State'] == r].plot(ax=ax, color=next(palette), label=r)
    
for i, point in gdf_regions.centroid.iteritems():
    reg_n = gdf_regions.loc[i, 'State']
    ax.text(s=reg_n, x=point.x-(len(reg_n)/6), y=point.y, fontsize='medium')

gdf_points[gdf_points['lat'] != -999].plot(ax=ax, markersize=5, color='red')
df_ind_pov['state_poverty_line'] = df_ind_pov['2011 rural percentage']/100 * df_ind_pov['Rural 2011-12 Poverty Expenditure Per Capita'].str.replace(',', '').astype(float) + (100-df_ind_pov['2011 rural percentage'])/100 * df_ind_pov['Urban 2011-12 Poverty Expenditure Per Capita'].str.replace(',', '').astype(float)
df_ind_pov.head()[['State', 'Rural 2011-12 Poverty Expenditure Per Capita', 'Urban 2011-12 Poverty Expenditure Per Capita', 'Headcount Ratio (%)', '2011 rural percentage', 'state_poverty_line']]
#do an india summary of loans and plot against state headcount percentage
gdf_sum = gdf_points[['State']].apply(pd.value_counts).reset_index()
gdf_sum.rename(index=str, columns={'State': 'loan_count'}, inplace=True)
gdf_sum.rename(index=str, columns={'index': 'State'}, inplace=True)
gdf_regions = gdf_regions.merge(gdf_sum, how='left', on='State')
gdf_regions['loan_count'] = gdf_regions['loan_count'].fillna(0)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 8), sharex=True)

county_order = gdf_regions.sort_values('Headcount Ratio (%)')['State']

sns.barplot(x='State', y='Headcount Ratio (%)', data=gdf_regions, ax=ax1, color='c', order=county_order)
sns.barplot(x='State', y='loan_count', data=gdf_regions, ax=ax2, color='r', order=county_order)

ax1.set_xlabel('')
ax1.set_title('India Headcount Ratio (% in poverty) vs. Loan Count by State', fontsize=15)

plt.xticks(rotation='vertical')

plt.show()
df_ind_pov['CFII Inverse'] = 1 - df_ind_pov['CFII']
gdf_regions['CFII Inverse'] = 1 - gdf_regions['CFII']
df_ind_pov.head()[['State', 'FII Rank', 'CFII', 'CFII Inverse', 'CDI']]
Blues = plt.get_cmap('Blues')

fig, ax = plt.subplots(1, figsize=(12,12))

regions = gdf_regions[~gdf_regions['CFII Inverse'].isnull()]['State']
for r in regions:
    gdf_regions[gdf_regions['State'] == r].plot(ax=ax, color=Blues(gdf_regions[gdf_regions['State'] == r]['CFII Inverse']*.65))
    
regions = gdf_regions[gdf_regions['CFII Inverse'].isnull()]['State']
for r in regions:
    gdf_regions[gdf_regions['State'] == r].plot(ax=ax, color='silver')

gdf_points[(gdf_points['lng'] != -999) & (gdf_points['Partner ID'] == 428)].plot(ax=ax, markersize=10, color='red', label=428)
gdf_points[(gdf_points['lng'] != -999) & (gdf_points['Partner ID'] == 334)].plot(ax=ax, markersize=10, color='yellow', label=334)
gdf_points[(gdf_points['lng'] != -999) & (gdf_points['Partner ID'] == 241)].plot(ax=ax, markersize=10, color='green', label=241)
gdf_points[(gdf_points['lng'] != -999) & (gdf_points['Partner ID'] == 347)].plot(ax=ax, markersize=10, color='orange', label=347)
gdf_points[(gdf_points['lng'] != -999) & (gdf_points['Partner ID'] == 242)].plot(ax=ax, markersize=10, color='pink', label=242)
gdf_points[(gdf_points['lng'] != -999) & (gdf_points['Partner ID'] == 238)].plot(ax=ax, markersize=10, color='lime', label=238)
gdf_points[(gdf_points['lng'] != -999) & (gdf_points['Partner ID'] == 225)].plot(ax=ax, markersize=10, color='purple', label=225)
gdf_points[(gdf_points['lng'] != -999) & (gdf_points['Partner ID'] == 212)].plot(ax=ax, markersize=10, color='darkviolet', label=212)
gdf_points[(gdf_points['lng'] != -999) & (gdf_points['Partner ID'] == 412)].plot(ax=ax, markersize=10, color='crimson', label=412)

ax.set_title('Kiva Loans in Indian States by Conditional Financial Exclusion\nDarker = Higher Financial Exclusion, Silver = No Financial Exclusion Data')

for i, point in gdf_regions.centroid.iteritems():
    reg_n = gdf_regions.loc[i, 'State']
    ax.text(s=reg_n, x=point.x-(len(reg_n)/6), y=point.y, fontsize='medium')

ax.legend(ncol=2, loc='best', frameon=True)
leg = ax.get_legend()
new_title = 'Partner ID'
leg.set_title(new_title)

plt.show()
gdf_cfii = gdf_points[~gdf_points['State'].isnull()].merge(gdf_regions[['State', 'CFII Inverse']], on='State')
gdf_cfii['cfii_weight'] = gdf_cfii['loan_amount'] * gdf_cfii['CFII Inverse']
gdf_cfii = gdf_cfii.groupby('Partner ID')[['loan_amount', 'cfii_weight']].sum().reset_index()
gdf_cfii['CFII Score'] = gdf_cfii['cfii_weight'] / gdf_cfii['loan_amount']
gdf_cfii['Partner ID'] = gdf_cfii['Partner ID'].astype(str).str.split('.', 1).str[0]

fig, ax = plt.subplots(1, 1, figsize=(10, 6))

prtnr_order = gdf_cfii.sort_values('CFII Score')['Partner ID']

sns.barplot(x='Partner ID', y='CFII Score', data=gdf_cfii, ax=ax, color='mediumpurple', order=prtnr_order)

ax.set_title('India Financial Exclusion Score by Partner ID', fontsize=15)

plt.show()
#make match happy
df_ind_cons['State'] = np.where(df_ind_cons['State'] == 'Orissa/Odisha', 'Odisha', df_ind_cons['State'])
df_ind_cons['State'] = np.where(df_ind_cons['State'] == 'Uttarkhand', 'Uttarakhand', df_ind_cons['State'])

#set consumption per capita
df_ind_cons['cons_per_capita'] = df_ind_cons['Rural Expenditure Per Capita'] * (df_ind_cons['2011 rural percentage'] / 100) + df_ind_cons['Urban Expenditure Per Capita'] * ((100 - df_ind_cons['2011 rural percentage']) / 100)
#normalize and take the inverse
max_value = df_ind_cons['cons_per_capita'].max()
min_value = df_ind_cons['cons_per_capita'].min()
df_ind_cons['cons_nrml'] = (df_ind_cons['cons_per_capita'] - min_value) / (max_value - min_value)
df_ind_cons['nrml_invrs_cons_per_capita'] = 1 - df_ind_cons['cons_nrml']

#add data to regions
gdf_regions = gdf_regions.merge(df_ind_cons[['State', 'cons_per_capita', 'nrml_invrs_cons_per_capita']], how='left', on='State')

#show data being combined
df_ind_cons.head()
Blues = plt.get_cmap('Blues')

fig, ax = plt.subplots(1, figsize=(12,12))

regions = gdf_regions[~gdf_regions['cons_per_capita'].isnull()]['State']
for r in regions:
    gdf_regions[gdf_regions['State'] == r].plot(ax=ax, color=Blues(gdf_regions[gdf_regions['State'] == r]['nrml_invrs_cons_per_capita']))
    
regions = gdf_regions[gdf_regions['cons_per_capita'].isnull()]['State']
for r in regions:
    gdf_regions[gdf_regions['State'] == r].plot(ax=ax, color='silver')

gdf_points[(gdf_points['lng'] != -999) & (gdf_points['Partner ID'] == 428)].plot(ax=ax, markersize=10, color='red', label=428)
gdf_points[(gdf_points['lng'] != -999) & (gdf_points['Partner ID'] == 334)].plot(ax=ax, markersize=10, color='yellow', label=334)
gdf_points[(gdf_points['lng'] != -999) & (gdf_points['Partner ID'] == 241)].plot(ax=ax, markersize=10, color='green', label=241)
gdf_points[(gdf_points['lng'] != -999) & (gdf_points['Partner ID'] == 347)].plot(ax=ax, markersize=10, color='orange', label=347)
gdf_points[(gdf_points['lng'] != -999) & (gdf_points['Partner ID'] == 242)].plot(ax=ax, markersize=10, color='pink', label=242)
gdf_points[(gdf_points['lng'] != -999) & (gdf_points['Partner ID'] == 238)].plot(ax=ax, markersize=10, color='lime', label=238)
gdf_points[(gdf_points['lng'] != -999) & (gdf_points['Partner ID'] == 225)].plot(ax=ax, markersize=10, color='purple', label=225)
gdf_points[(gdf_points['lng'] != -999) & (gdf_points['Partner ID'] == 212)].plot(ax=ax, markersize=10, color='darkviolet', label=212)
gdf_points[(gdf_points['lng'] != -999) & (gdf_points['Partner ID'] == 412)].plot(ax=ax, markersize=10, color='crimson', label=412)

ax.set_title('Kiva Loans in Indian States by Inverse Normalized Consumption Per Capita\nDarker = Lower Consumption, Silver = No Consumption Data')

for i, point in gdf_regions.centroid.iteritems():
    reg_n = gdf_regions.loc[i, 'State']
    ax.text(s=reg_n, x=point.x-(len(reg_n)/6), y=point.y, fontsize='medium')

ax.legend(ncol=2, loc='best', frameon=True)
leg = ax.get_legend()
new_title = 'Partner ID'
leg.set_title(new_title)

plt.show()
gdf_ind_cons = gdf_points[~gdf_points['State'].isnull()].merge(gdf_regions[['State', 'nrml_invrs_cons_per_capita']], on='State')
gdf_ind_cons['cons_weight'] = gdf_ind_cons['loan_amount'] * gdf_ind_cons['nrml_invrs_cons_per_capita']
gdf_ind_cons = gdf_ind_cons.groupby('Partner ID')[['loan_amount', 'cons_weight']].sum().reset_index()
gdf_ind_cons['Consumption Score'] = gdf_ind_cons['cons_weight'] / gdf_ind_cons['loan_amount']
gdf_ind_cons['Partner ID'] = gdf_ind_cons['Partner ID'].astype(str).str.split('.', 1).str[0]
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

region_order = gdf_ind_cons.sort_values('Consumption Score')['Partner ID']

sns.barplot(x='Partner ID', y='CFII Score', data=gdf_cfii, ax=ax1, color='mediumpurple', order=region_order)
sns.barplot(x='Partner ID', y='Consumption Score', data=gdf_ind_cons, ax=ax2, color='teal', order=region_order)

ax1.set_xlabel('')
ax1.set_title('India CFII Score vs. Consumption Score by Partner ID', fontsize=15)

plt.show()
ISO = 'RWA'
epsg = '1199'

### POLYGONS ###
# read in geospatial regions
gdf_rwa = gpd.GeoDataFrame.from_file("../input/rwanda-2006-geospatial-administrative-regions/Province_Boundary_2006.shp")

gdf_rwa['Prov_Name'] = np.where(gdf_rwa['Prov_Name'] == 'Southern Province', 'South', gdf_rwa['Prov_Name'])
gdf_rwa['Prov_Name'] = np.where(gdf_rwa['Prov_Name'] == 'Western Province', 'West', gdf_rwa['Prov_Name'])
gdf_rwa['Prov_Name'] = np.where(gdf_rwa['Prov_Name'] == 'Eastern Province', 'East', gdf_rwa['Prov_Name'])
gdf_rwa['Prov_Name'] = np.where(gdf_rwa['Prov_Name'] == 'Northern Province', 'North', gdf_rwa['Prov_Name'])
gdf_rwa = gdf_rwa.rename(index=str, columns={'Prov_Name': 'mpi_region_new'})
gdf_rwa.crs = {"init":epsg}

gdf_rwa = gdf_rwa.merge(df_mpi_subntl[df_mpi_subntl['ISO country code'] == ISO][['Sub-national region', 'MPI Regional']], how='left', left_on='mpi_region_new', right_on='Sub-national region')


#loan themes
gdf_regions = gdf_rwa
gdf_points = gdf_loan_theme[gdf_loan_theme['ISO'] == ISO]

### POINTS IN POLYGONS
for i in range(0, len(gdf_regions)):
    print('i is: ' + str(i) + ' at ' + strftime("%Y-%m-%d %H:%M:%S", gmtime()))
    gdf_points['r_map'] = gdf_points.within(gdf_regions['geometry'][i])
    gdf_points['mpi_region_new'] = np.where(gdf_points['r_map'], gdf_regions['mpi_region_new'][i], gdf_points['mpi_region_new'])
# amended sub-national partner regions
LTsubnat = gdf_points[~(gdf_points['mpi_region_new'] == 'nan')]
LTsubnat = LTsubnat.merge(gdf_regions[['mpi_region_new', 'MPI Regional']], on='mpi_region_new')

#~ Get total volume and average MPI Regional Score for each partner loan theme
LS = LTsubnat.groupby(['Partner ID', 'ISO']).agg({'MPI Regional': np.mean, 'amount': np.sum, 'number': np.sum})
#~ Get a volume-weighted average of partners loanthemes.
weighted_avg_LTsubnat = lambda df: np.average(df['MPI Regional'], weights=df['amount'])
#~ and get weighted average for partners. 
MPI_reg_reassign = LS.groupby(level=['Partner ID', 'ISO']).apply(weighted_avg_LTsubnat)
MPI_reg_reassign = MPI_reg_reassign.to_frame()
MPI_reg_reassign.reset_index(level=1, inplace=True)
MPI_reg_reassign = MPI_reg_reassign.rename(index=str, columns={0: 'MPI Score'})

MPI_reg_reassign[MPI_reg_reassign['ISO'] == ISO].reset_index()
# now let's do loans
gdf_points = gdf_kv_loans[gdf_kv_loans['country'] == 'Rwanda']
gdf_points = gdf_points.rename(index=str, columns={'partner_id': 'Partner ID'})

### POINTS IN POLYGONS
# commented out for speed sake of me working on my own work in this kernel
for i in range(0, len(gdf_regions)):
    print('i is: ' + str(i) + ' at ' + strftime("%Y-%m-%d %H:%M:%S", gmtime()))
    gdf_points['r_map'] = gdf_points.within(gdf_regions['geometry'][i])
    gdf_points['mpi_region_new'] = np.where(gdf_points['r_map'], gdf_regions['mpi_region_new'][i], gdf_points['mpi_region_new'])
gdf_points[['id', 'mpi_region_new']].to_csv(ISO + '_loans_mpi_pip_output.csv', index = False)
# amended sub-national loans - average
LTsubnat = gdf_points[~(gdf_points['mpi_region_new'] == 'nan') & ~(gdf_points['mpi_region_new'].isnull())]
LTsubnat = LTsubnat.merge(gdf_regions[['mpi_region_new', 'MPI Regional']], on='mpi_region_new')

LTsubnat['number'] = 1

#~ Get total volume and average MPI Regional Score for each partner loan theme
LS = LTsubnat.groupby(['Partner ID']).agg({'MPI Regional': np.mean, 'loan_amount': np.sum, 'number': np.sum})
#~ Get a volume-weighted average of partners loanthemes.
weighted_avg_LTsubnat = lambda df: np.average(df['MPI Regional'], weights=df['loan_amount'])
#~ and get weighted average for partners. 
MPI_loan_reassign = LS.groupby(level=['Partner ID']).apply(weighted_avg_LTsubnat)
MPI_loan_reassign = MPI_loan_reassign.to_frame()
MPI_loan_reassign.reset_index(level=0, inplace=True)
MPI_loan_reassign = MPI_loan_reassign.rename(index=str, columns={0: 'MPI Score'})

MPI_loan_reassign
# amended sub-national loans - weighted
gdf_out = gdf_points[~(gdf_points['mpi_region_new'] == 'nan')].merge(gdf_regions[['mpi_region_new', 'MPI Regional']], on='mpi_region_new')
gdf_out['MPI Weight'] = gdf_out['loan_amount'] * gdf_out['MPI Regional']
gdf_out = gdf_out.groupby('Partner ID')[['loan_amount', 'MPI Weight']].sum().reset_index()
gdf_out['MPI Score'] = gdf_out['MPI Weight'] / gdf_out['loan_amount']
gdf_out[['Partner ID', 'MPI Score']]
Blues = plt.get_cmap('Blues')

regions = gdf_regions['mpi_region_new']

fig, ax = plt.subplots(1, figsize=(12,12))

for r in regions:
    gdf_regions[gdf_regions['mpi_region_new'] == r].plot(ax=ax, color=Blues(gdf_regions[gdf_regions['mpi_region_new'] == r]['MPI Regional']*2))

gdf_points[(gdf_points['lng'] != -999) & (gdf_points['Partner ID'] == 161)].plot(ax=ax, markersize=10, color='red', label=161)
gdf_points[(gdf_points['lng'] != -999) & (gdf_points['Partner ID'] == 117)].plot(ax=ax, markersize=10, color='yellow', label=117)
gdf_points[(gdf_points['lng'] != -999) & (gdf_points['Partner ID'] == 441)].plot(ax=ax, markersize=10, color='green', label=441)
gdf_points[(gdf_points['lng'] != -999) & (gdf_points['Partner ID'] == 271)].plot(ax=ax, markersize=10, color='orange', label=271)
gdf_points[(gdf_points['lng'] != -999) & (gdf_points['Partner ID'] == 319)].plot(ax=ax, markersize=10, color='pink', label=319)
gdf_points[(gdf_points['lng'] != -999) & (gdf_points['Partner ID'] == 528)].plot(ax=ax, markersize=10, color='lime', label=528)
gdf_points[(gdf_points['lng'] != -999) & (gdf_points['Partner ID'] == 493)].plot(ax=ax, markersize=10, color='purple', label=493)
gdf_points[(gdf_points['lng'] != -999) & (gdf_points['Partner ID'] == 212)].plot(ax=ax, markersize=10, color='darkviolet', label=212)


for i, point in gdf_regions.centroid.iteritems():
    #reg_n = gdf_regions.iloc[i]['mpi_region_new']
    reg_n = gdf_regions.loc[i, 'mpi_region_new']
    ax.text(s=reg_n, x=point.x-.1, y=point.y, fontsize='large')
    

ax.set_title('Loans across Rwanda by Field Partner\nDarker = Higher MPI.  Range from 0.118 (Kigali City) to 0.295 (South)')
ax.legend(loc='upper left', frameon=True)
leg = ax.get_legend()
new_title = 'Partner ID'
leg.set_title(new_title)

plt.show()
rwanda_only_partners = ['161', '117', '441', '271', '528', '212']

# national scores
df_curr_ntl_mpi = LT[LT['ISO'] == ISO][['Partner ID', 'MPI Score']].drop_duplicates()
df_curr_ntl_mpi['method'] = 'Existing National (rural_percentage)'

# subnational scores
df_curr_subnat = MPI_regional_scores[MPI_regional_scores['ISO'] == ISO].reset_index()
df_curr_subnat = df_curr_subnat[['Partner ID', 'MPI Score']]
df_curr_subnat['method'] = 'Existing Sub-National (Partner Regions)'

# amended region theme scores
df_amend_fld_rgn = MPI_reg_reassign[MPI_reg_reassign['ISO'] == ISO].reset_index()[['Partner ID', 'MPI Score']]
df_amend_fld_rgn['method'] = 'Amended Sub-National (Partner Regions)'

# amended loan scores - averaged
MPI_loan_reassign['method'] = 'Amended Sub-National (Loans - Mean)'

# amended loan scores - weighted
gdf_out = gdf_out[['Partner ID', 'MPI Score']]
gdf_out['method'] = 'Amended Sub-National (Loans - Weighted)'

# combine for comparison
frames = (df_curr_ntl_mpi, df_curr_subnat, df_amend_fld_rgn, MPI_loan_reassign, gdf_out)
df_compare = pd.concat(frames)
df_compare['Partner ID'] = df_compare['Partner ID'].astype(str).str.split('.', 1).str[0]
df_compare = df_compare[df_compare['Partner ID'].isin(rwanda_only_partners)]
fig, ax = plt.subplots(figsize=(20, 10))
sns.set_palette('muted')
sns.barplot(x='Partner ID', y='MPI Score', data=df_compare, hue='method')

ax.legend(ncol=1, loc='best', frameon=True)
ax.set(ylabel='MPI Score',
       xlabel='Partner ID')

leg = ax.get_legend()
new_title = 'Method'
leg.set_title(new_title)

ax.set_title('Existing vs. Amended Field Partner MPI - Kenya', fontsize=15)
plt.show()
df_rwa_bank.head()
gdf_rwa_dist = gpd.GeoDataFrame.from_file("../input/rwanda-2006-geospatial-administrative-regions/District_Boundary_2006.shp")
gdf_rwa_dist = gdf_rwa_dist.rename(index=str, columns={'District': 'district'})
gdf_rwa_dist.crs = {"init":epsg}

df_rwa_bank['excluded_pct'] = df_rwa_bank['Excluded'] / 100.0
gdf_rwa_dist = gdf_rwa_dist.merge(df_rwa_bank, how='left', on='district')

gdf_regions = gdf_rwa_dist

gdf_points['district'] = np.NaN
for i in range(0, len(gdf_regions)):
    print('i is: ' + str(i) + ' at ' + strftime("%Y-%m-%d %H:%M:%S", gmtime()))
    gdf_points['r_map'] = gdf_points.within(gdf_regions['geometry'][i])
    gdf_points['district'] = np.where(gdf_points['r_map'], gdf_regions['district'][i], gdf_points['district'])
gdf_points[['id', 'district']].to_csv(ISO + '_loans_dist_pip_output.csv', index = False)
#do an rwanda summary of loans and plot against state headcount percentage
gdf_sum = gdf_points[['district']].apply(pd.value_counts).reset_index()
gdf_sum.rename(index=str, columns={'district': 'loan_count'}, inplace=True)
gdf_sum.rename(index=str, columns={'index': 'district'}, inplace=True)
gdf_regions = gdf_regions.merge(gdf_sum, how='left', on='district')
gdf_regions['loan_count'] = gdf_regions['loan_count'].fillna(0)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 8), sharex=True)

county_order = gdf_regions.sort_values('excluded_pct')['district']

sns.barplot(x='district', y='excluded_pct', data=gdf_regions, ax=ax1, color='c', order=county_order)
sns.barplot(x='district', y='loan_count', data=gdf_regions, ax=ax2, color='r', order=county_order)

ax1.set_xlabel('')
ax1.set_title('Rwanda Percentage Financially Excluded vs. Loan Count by District', fontsize=15)

plt.xticks(rotation='vertical')

plt.show()
Blues = plt.get_cmap('Blues')

regions = gdf_regions['district']

fig, ax = plt.subplots(1, figsize=(12,12))

for r in regions:
    gdf_regions[gdf_regions['district'] == r].plot(ax=ax, color=Blues(gdf_regions[gdf_regions['district'] == r]['excluded_pct']*3.5))

gdf_points[(gdf_points['lng'] != -999) & (gdf_points['Partner ID'] == 161)].plot(ax=ax, markersize=10, color='red', label=161)
gdf_points[(gdf_points['lng'] != -999) & (gdf_points['Partner ID'] == 117)].plot(ax=ax, markersize=10, color='yellow', label=117)
gdf_points[(gdf_points['lng'] != -999) & (gdf_points['Partner ID'] == 441)].plot(ax=ax, markersize=10, color='green', label=441)
gdf_points[(gdf_points['lng'] != -999) & (gdf_points['Partner ID'] == 271)].plot(ax=ax, markersize=10, color='orange', label=271)
gdf_points[(gdf_points['lng'] != -999) & (gdf_points['Partner ID'] == 319)].plot(ax=ax, markersize=10, color='pink', label=319)
gdf_points[(gdf_points['lng'] != -999) & (gdf_points['Partner ID'] == 528)].plot(ax=ax, markersize=10, color='lime', label=528)
gdf_points[(gdf_points['lng'] != -999) & (gdf_points['Partner ID'] == 493)].plot(ax=ax, markersize=10, color='purple', label=493)
gdf_points[(gdf_points['lng'] != -999) & (gdf_points['Partner ID'] == 212)].plot(ax=ax, markersize=10, color='darkviolet', label=212)


for i, point in gdf_regions.centroid.iteritems():
    reg_n = gdf_regions.loc[i, 'district']
    ax.text(s=reg_n, x=point.x-.1, y=point.y, fontsize='large')
    

ax.set_title('Loans across Rwanda by Field Partner by District\nDarker = Higher Financial Exclusion Percentage')
ax.legend(loc='upper left', frameon=True)
leg = ax.get_legend()
new_title = 'Partner ID'
leg.set_title(new_title)

plt.show()
gdf_excl = gdf_points[~gdf_points['district'].isnull()].merge(gdf_regions[['district', 'excluded_pct']], on='district')
gdf_excl['excl_weight'] = gdf_excl['loan_amount'] * gdf_excl['excluded_pct']
gdf_excl = gdf_excl.groupby('Partner ID')[['loan_amount', 'excl_weight']].sum().reset_index()
gdf_excl['Exclusion Score'] = gdf_excl['excl_weight'] / gdf_excl['loan_amount']
gdf_excl['Partner ID'] = gdf_excl['Partner ID'].astype(str).str.split('.', 1).str[0]
gdf_excl = gdf_excl.merge(df_compare[df_compare['method'] == 'Amended Sub-National (Loans - Weighted)'], how='left', on='Partner ID')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

region_order = gdf_excl.sort_values('Exclusion Score')['Partner ID']

sns.barplot(x='Partner ID', y='MPI Score', data=gdf_excl, ax=ax1, color='deepskyblue', order=region_order)
sns.barplot(x='Partner ID', y='Exclusion Score', data=gdf_excl, ax=ax2, color='mediumorchid', order=region_order)

ax1.set_xlabel('')
ax1.set_title('Rwanda MPI Score vs. Exclusion Score by Partner ID', fontsize=15)

plt.show()
#df_cato.head()
df_cato[df_cato['ISO_Code'].isin(['MOZ', 'PHL', 'KEN'])]
def read_findex(datafile=None, interpolate=False, invcov=True, variables = ["Account", "Loan", "Emergency"], norm=True):
    """
    Returns constructed findex values for each country

    Read in Findex data - Variables include: Country ISO Code, Country Name,
                          Pct with Account at Financial institution (Poor),
                          Pct with a loan from a Financial institution (Poor),
                          Pct who say they could get an emergency loan (Poor)

    Take average of 'poorest 40%' values for each value in `variables'

     If `normalize':
        Apply the normalization function to every MPI variable
    """
    if datafile == None: datafile = "../input/findex-world-bank/FINDEXData.csv"

    F = pd.read_csv(datafile)#~ [["ISO","Country Name", "Indicator Name", "MRV"]]
    
    Fcols = {'Country Name': 'Country',
        'Country Code': 'ISO',
        'Indicator Name': 'indicator',
        'Indicator Code': 'DROP',
        '2011': 'DROP',
        '2014': 'DROP',
        'MRV': 'Val'
        }
    F = F.rename(columns=Fcols).drop("DROP",1)
    F['Val'] /= 100.
    
    indicators = {"Account at a financial institution, income, poorest 40% (% ages 15+) [ts]": "Account",
        "Coming up with emergency funds: somewhat possible, income, poorest 40% (% ages 15+) [w2]": "Emergency",
        "Coming up with emergency funds: very possible, income, poorest 40% (% ages 15+) [w2]": "Emergency",
        "Borrowed from a financial institution, income, poorest 40% (% ages 15+) [ts]": "Loan"
        }

    F['Poor'] = F['indicator'].apply(lambda ind: "Poor" if "poorest" in ind else "Rich") 
    F['indicator'] = F['indicator'].apply(lambda ind: indicators.setdefault(ind,np.nan)) 
    F = F.dropna(subset=["indicator"])
    F = F.groupby(["Poor","ISO","indicator"])["Val"].sum()
    F = 1 - F.loc["Poor"]

    F = F.unstack("indicator")
    
    # fill missing values for the emergency indicator with a predicted score from OLS regression analysis 
    if interpolate:
        results = smf.ols("Emergency ~ Loan + Account",data=F).fit()
        F['Emergency_fit'] = results.params['Intercept'] + F[['Loan','Account']].mul(results.params[['Loan','Account']]).sum(1)
        F['Emergency'].fillna(F['Emergency_fit'],inplace=True)
    if invcov: F['Findex'] = invcov_index(F[variables]) #.mean(1)
    else: F['Findex'] = F[variables].mean(1,skipna=True)
        
    flatvar = flatten(F['Findex'].dropna(), use_buckets = False, return_buckets = False)
    F = F.join(flatvar,how='left',lsuffix=' (raw)')
    
    return F

def invcov_index(indicators):
    """
    Convert a dataframe of indicators into an inverse covariance matrix index
    """
    df = indicators.copy()
    df = (df-df.mean())/df.std()
    I  = np.ones(df.shape[1])
    E  = inv(df.cov())
    s1  = I.dot(E).dot(I.T)
    s2  = I.dot(E).dot(df.T)
    try:
        int(s1)
        S  = s2/s1
    except TypeError: 
        S  = inv(s1).dot(s2)
    
    S = pd.Series(S,index=indicators.index)

    return S

def flatten(Series, outof = 10., bins = 20, use_buckets = False, write_buckets = False, return_buckets = False):
    """
    NOTE: Deal with missing values, obviously!
    Convert Series to a uniform distribution from 0 to `outof'
    use_buckets uses the bucketing rule from a previous draw.
    """

    tempSeries = Series.dropna()
    if use_buckets: #~ Use a previously specified bucketing rule
        cuts, pcts = list(rule['Buckets']), np.array(rule['Values']*(100./outof))
    else: #~ Make Bucketing rule to enforce a uniform distribution
        pcts = np.append(np.arange(0,100,100/bins),[100])
        cuts = [ np.percentile(tempSeries,p) for p in pcts ]
        while len(cuts)>len(set(cuts)):
            bins -= 1
            pcts = np.append(np.arange(0,100,100/bins),[100])
            cuts = [ np.percentile(tempSeries,p) for p in pcts ]

    S = pd.cut(tempSeries,cuts,labels = pcts[1:]).astype(float)
    S *= outof/100

    buckets = pd.DataFrame({"Buckets":cuts,"Values":pcts*(outof/100)})

    if return_buckets: return S, 
    else: return S
    
F = read_findex()
df_findex = F.reset_index()
#df_findex.head()
df_findex[df_findex['ISO'].isin(['MOZ', 'PHL', 'KEN'])]
Fcol = ['Country Code', 'Indicator Name', 'MRV']
F = pd.read_csv("../input/findex-world-bank/FINDEXData.csv", usecols=Fcol)

indicators_search = 'Borrowed any money in the past year|\
Borrowed for education or school fees|\
Borrowed for health or medical purposes|\
Borrowed from a financial institution|\
Borrowed from a private informal lender|\
Borrowed from family or friends|\
Borrowed to start, operate, or expand a farm or business|\
Coming up with emergency funds: not at all possible|\
Coming up with emergency funds: not very possible|\
Coming up with emergency funds: somewhat possible|\
Coming up with emergency funds: very possible|\
Account at a financial institution|\
Saved at a financial institution|\
Main source of emergency funds: family or friends|\
Main source of emergency funds: savings'
# Loan in the past year

group_search_income = 'richest|poorest'

def clean(findex, indicators_search, group_search=group_search_income):
    """Split Indicator Name into two columns: Indicator and Group."""

    # select only rows from wave 2 (2014) with the following indicators and groups
    findex = findex[findex['Indicator Name'].str.contains('w2')]
    findex = findex[findex['Indicator Name'].str.contains(indicators_search)]
    findex = findex[findex['Indicator Name'].str.contains(group_search)]

    # remove last words from the string
    findex['Indicator Name'] = findex['Indicator Name'].str.split().str[:-4]
    findex['Indicator Name'] = findex['Indicator Name'].apply(lambda x: ' '.join(x))

    # create a column for the indicator and one for the income group
    findex['Group'] = findex['Indicator Name'].str.split().str[-2:]
    findex['Group'] = findex['Group'].apply(lambda x: ' '.join(x))

    findex['Indicator'] = findex['Indicator Name'].str.split().str[:-3]
    findex['Indicator'] = findex['Indicator'].apply(lambda x: ' '.join(x))
    # remove last words from the string
    findex['Indicator'] = findex['Indicator'].str[:-1]

    # remove column Indicator Name
    findex.drop('Indicator Name', axis=1, inplace=True)

    # set index
    findex.set_index(['Country Code', 'Indicator', 'Group'], inplace=True)

    # create series
    findex = findex['MRV']

    # unstack series
    findex = findex.unstack(level=-1)
    
    return findex

def add_formal_informal_borrowing(df):
    
    # df['Informal'] = df['Borrowed from a private informal lender'] + df['Borrowed from family or friends']
    df['Informal'] = df['Borrowed from a private informal lender']
    df['Informal ratio'] = df['Informal']/df['Borrowed any money in the past year']
    df['Formal ratio'] = df['Borrowed from a financial institution']/df['Borrowed any money in the past year']
    df['Formal/Informal ratio'] = df['Formal ratio']/df['Informal ratio'] 
       
    return df

def compute_finexcl(df):
    RF = df.loc[df['Group'] == 'richest 60%', 'Formal ratio']
    RI = df.loc[df['Group'] == 'richest 60%', 'Informal ratio']
    PF = df.loc[df['Group'] == 'poorest 40%', 'Formal ratio']
    PI = df.loc[df['Group'] == 'poorest 40%', 'Informal ratio']
    finexcl = (float(RF-RI)-float(PF-PI))/float(RF-RI)

    df['Financial Exclusion'] = finexcl
        
    return df

def main():
    
    # return a dataframe with groups as variables
    # merge findex percentages and findex index
    F_groups = clean(F, indicators_search, group_search=group_search_income)
    
    # return a dataframe with the indicators as variables
    # add variables for formal and informal borrowing to dataframe
    F_ind = F_groups.stack().unstack(level=1)
    F_ind = add_formal_informal_borrowing(F_ind)
        
    # add indicator for financial exclusion to dataframe
    F_ind.reset_index(inplace=True)
    F_ind.rename(columns={'level_1':'Group'}, inplace=True)
    
    grouped = F_ind.groupby('Country Code')
    F_ind_copy = pd.DataFrame()
    for user, group in grouped:
        group = compute_finexcl(group)
        F_ind_copy = F_ind_copy.append(group)
    F_ind_copy.set_index(['Country Code', 'Group'], inplace=True)
    F_ind = F_ind_copy
    
    
    print ("2 Dataframes returned: findex_groups (where the variables are the groups) and findex_ind(where the indicators are the variables).")
    
    return F_groups, F_ind

F_groups, F_ind = main()

F_ind = F_ind[['Informal', 'Formal ratio', 'Informal ratio', 'Financial Exclusion']]

df_findex2 = F_ind.reset_index()
df_findex2[df_findex2['Country Code'].isin(['MOZ', 'PHL', 'KEN'])]
df_indices = df_mpi_subntl[['ISO country code', 'MPI National']].drop_duplicates().merge(df_findex[['ISO', 'Findex (raw)', 'Findex']], how='left', left_on='ISO country code', right_on='ISO')
df_indices = df_indices.merge(df_findex2, how='left', left_on='ISO country code', right_on='Country Code')
df_indices[df_indices['ISO country code'].isin(['MOZ', 'PHL', 'KEN'])].sort_values(['ISO country code', 'Group'])
# get the MPI values
df_out = df_mpi_ntl[['ISO', 'MPI Rural', 'MPI Urban']].merge(df_wb_rural[['ISO', '2016']], how='left', on='ISO')
df_out = df_out.merge(df_mpi_subntl[['ISO country code', 'MPI National']].drop_duplicates(), how='left', left_on='ISO', right_on='ISO country code')
df_out['MPI National Calculated'] = df_out['2016'] / 100 * df_out['MPI Rural'] + (100 - df_out['2016']) / 100 * df_out['MPI Urban']
#df_out.merge(df_mpi_subntl[['ISO country code', 'MPI National']].drop_duplicates(), how='left', left_on='ISO', right_on='ISO country code')
df_out['MPI National'] = np.where(df_out['MPI National'].isnull(), df_out['MPI National Calculated'], df_out['MPI National'])

# merge with cato values
df_out = df_cato[['ISO_Code', 'Countries', 'Sound Money', 'Credit market regulations', 'Business regulations', 'ECONOMIC FREEDOM (Score)', 'Income per Capita']].merge(df_out, how='outer', left_on='ISO_Code', right_on='ISO')
df_out['ISO'] = np.where(df_out['ISO'].isnull(), df_out['ISO_Code'], df_out['ISO'])
df_out = df_out[['ISO', 'Countries', 'Sound Money', 'Credit market regulations', 'Business regulations', 'ECONOMIC FREEDOM (Score)', 'Income per Capita', 'MPI National']]

# merge with findex and financial exclusion
df_out = df_out.merge(df_findex[['ISO', 'Findex']], how='outer', on='ISO')
df_out = df_out.merge(df_findex2[['Country Code', 'Financial Exclusion']].drop_duplicates(), how='outer', left_on='ISO', right_on='Country Code')

# i don't like Kazakhstan's 0 MPI
df_out = df_out[df_out['ISO'] != 'KAZ']

# add in consumption data.  most populated for 2010
df_wb_cons = df_wb_cons.rename(index=str, columns={'2010': 'cons_per_capita'})
df_out = df_out.merge(df_wb_cons[['Country Code', 'cons_per_capita']], left_on='ISO', right_on='Country Code')
df_pair = df_out[['ISO', 'Countries', 'Credit market regulations', 'ECONOMIC FREEDOM (Score)', 'Income per Capita', 'cons_per_capita', 'Findex', 'Financial Exclusion']].dropna(axis=0, how='any')
g = sns.pairplot(df_pair[['Credit market regulations', 'ECONOMIC FREEDOM (Score)', 'Income per Capita', 'cons_per_capita', 'Findex', 'Financial Exclusion']])
g.map_upper(sns.kdeplot)
g.map_diag(plt.hist)
g.map_lower(sns.regplot, scatter_kws={'alpha':0.20}, line_kws={'color': 'red'})
g.map_lower(corrfunc)
plt.show()
df_pair = df_out[['ISO', 'Countries', 'Credit market regulations', 'ECONOMIC FREEDOM (Score)', 'Income per Capita', 'cons_per_capita', 'MPI National', 'Findex', 'Financial Exclusion']].dropna(axis=0, how='any')
g = sns.pairplot(df_pair[['Credit market regulations', 'ECONOMIC FREEDOM (Score)', 'Income per Capita', 'cons_per_capita', 'MPI National', 'Findex', 'Financial Exclusion']])
g.map_upper(sns.kdeplot)
g.map_diag(plt.hist)
g.map_lower(sns.regplot, scatter_kws={'alpha':0.20}, line_kws={'color': 'red'})
g.map_lower(corrfunc)
plt.show()
#df_kv_loans.drop(columns=['ISO'], inplace=True)
df_kv_loans = df_kv_loans.merge(df_add_cntry[['country_code', 'country_code3']], how='left', on='country_code')
df_kv_loans = df_kv_loans.rename(index=str, columns={'country_code3': 'ISO'})

df_kv_loans['ISO'] = np.where(df_kv_loans['country_code'] == 'IQ', 'IRQ', df_kv_loans['ISO'])
df_kv_loans['ISO'] = np.where(df_kv_loans['country_code'] == 'CL', 'CHL', df_kv_loans['ISO'])
df_kv_loans['ISO'] = np.where(df_kv_loans['country_code'] == 'XK', 'XKX', df_kv_loans['ISO'])
df_kv_loans['ISO'] = np.where(df_kv_loans['country_code'] == 'CG', 'COG', df_kv_loans['ISO'])
df_kv_loans['ISO'] = np.where(df_kv_loans['country_code'] == 'MR', 'MRT', df_kv_loans['ISO'])
df_kv_loans['ISO'] = np.where(df_kv_loans['country_code'] == 'VU', 'VUT', df_kv_loans['ISO'])
df_kv_loans['ISO'] = np.where(df_kv_loans['country_code'] == 'PA', 'PAN', df_kv_loans['ISO'])
df_kv_loans['ISO'] = np.where(df_kv_loans['country_code'] == 'VI', 'VIR', df_kv_loans['ISO'])
df_kv_loans['ISO'] = np.where(df_kv_loans['country_code'] == 'VC', 'VCT', df_kv_loans['ISO'])
df_kv_loans['ISO'] = np.where(df_kv_loans['country_code'] == 'GU', 'GUM', df_kv_loans['ISO'])
df_kv_loans['ISO'] = np.where(df_kv_loans['country_code'] == 'PR', 'PRI', df_kv_loans['ISO'])
df_kv_loans['ISO'] = np.where(df_kv_loans['country_code'] == 'CI', 'CIV', df_kv_loans['ISO'])

df_sum = df_kv_loans.groupby(['ISO', 'country']).size().reset_index(name='counts')
df_sum = df_sum.merge(df_out, how='right', on='ISO')[['ISO', 'country', 'counts', 'Credit market regulations', 'ECONOMIC FREEDOM (Score)', 'Income per Capita', 'cons_per_capita', 'MPI National', 'Findex', 'Financial Exclusion']]

#manual massagey
df_sum['ISO'] = np.where(df_sum['ISO'] == 'PSE', 'PS', df_sum['ISO'])
df_sum['counts'] = df_sum['counts'].fillna(0)

df_sum = df_sum.merge(df_add_cntry[['country_code3', 'country_name']], how='left', left_on='ISO', right_on='country_code3')
df_sum['country'] = np.where(df_sum['country'].isnull(), df_sum['country_name'], df_sum['country'])
df_sum = df_sum[(~df_sum['country'].isnull()) & (~df_sum['Findex'].isnull())][['ISO', 'country', 'counts', 'Credit market regulations', 'ECONOMIC FREEDOM (Score)', 'Income per Capita', 'cons_per_capita', 'MPI National', 'Findex', 'Financial Exclusion']]
df_sum = df_sum[df_sum['cons_per_capita'] < 20000]
df_sum[(df_sum['counts'] < 50) & (df_sum['Findex'] > 5)].sort_values(['Findex', 'MPI National'], ascending=[False, False]).head(15)
df_sum[(df_sum['counts'] < 50) & (df_sum['Financial Exclusion'] < 2)].sort_values(['Financial Exclusion', 'MPI National'], ascending=[True, False]).head(15)
df_pri = df_sum[~df_sum['ISO'].isin(['AFG', 'BHR', 'BLR', 'COM', 'CZE', 'ESP', 'GRC', 'HKG', 'IRQ', 'KOR', 'KWT', 'MLT', 'NZL', 'OMN', 'PRI', 'PRT', 'PS', 'QAT', 'SAU', 'SDN', 'SGP', 'SVK', 'SVN', 'TKM', 'TTO', 'UZB', 'XKX'])]
df_pri = df_pri.merge(df_add_cntry[['country_code3', 'region']].drop_duplicates(), left_on='ISO', right_on='country_code3')

#economic freedom normalization
max_value = df_pri['ECONOMIC FREEDOM (Score)'].max()
min_value = df_pri['ECONOMIC FREEDOM (Score)'].min()
df_pri['economic_nrml'] = (df_pri['ECONOMIC FREEDOM (Score)'] - min_value) / (max_value - min_value)

#consumption per capita normalization
max_value = df_pri['cons_per_capita'].max()
min_value = df_pri['cons_per_capita'].min()
df_pri['cons_nrml'] = (df_pri['cons_per_capita'] - min_value) / (max_value - min_value)
df_pri['cons_nrml_inv'] = 1 - df_pri['cons_nrml']

#findex normalization
max_value = df_pri['Findex'].max()
min_value = df_pri['Findex'].min()
df_pri['findex_nrml'] = (df_pri['Findex'] - min_value) / (max_value - min_value)

#calculate priority index
df_pri['priority_idx'] = (df_pri['economic_nrml'] + df_pri['cons_nrml_inv'] + df_pri['findex_nrml']) / 3

#calculate loans per capita
df_pri = df_pri.merge(df_wb_pop[['Country Code', '2016']], left_on='ISO', right_on='Country Code')
df_pri['loans_per_cap'] = df_pri['counts'] / df_pri['2016']

df_pri.sort_values('priority_idx', ascending=False)[['ISO', 'country', 'counts', 'MPI National', 'ECONOMIC FREEDOM (Score)', 'cons_per_capita', 'Findex', 'priority_idx']].head()
fig, ax = plt.subplots(1, 1, figsize=(15, 9), sharex=True)
df_pri.sort_values('priority_idx', inplace=True, ascending=False)
sns.barplot(y=df_pri['country'].head(40), x=df_pri['priority_idx'].head(40), color='c')
ax.set_xlabel('priority index')
ax.set_ylabel('country')
ax.set_title('Countries by Priority of Impact (Top 40)', fontsize=15)
plt.show()
df_pri[df_pri['ISO'].isin(['KEN', 'RWA', 'COD', 'BDI', 'UGA'])][['ISO', 'country', 'counts', 'MPI National', 'Income per Capita', 'ECONOMIC FREEDOM (Score)', 'cons_per_capita', 'Findex', 'Financial Exclusion', 'priority_idx']]
trace = go.Scatter(
    x= df_pri['priority_idx'],
    y= df_pri['counts'],
    mode = 'markers',
    marker = dict(
        size = 10,
        opacity= 0.5
    ),
    text = df_pri['country']
)

layout = go.Layout(
    title= 'Countries by Priority Index and Loan Count',
    hovermode= 'closest',
    xaxis= dict(
        title= 'Priority Index (increasing = higher priority)',
        ticklen= 5,
        zeroline= False,
        gridwidth= 2,
    ),
    yaxis=dict(
        title= 'Loan Count',
        ticklen= 5,
        gridwidth= 2
    ),
    showlegend= False
)
fig = go.Figure(data=[trace], layout=layout)
py.iplot(fig)
trace0 = go.Scatter(
    x= df_pri[df_pri['region'] == 'Northern Africa']['priority_idx'],
    y= df_pri[df_pri['region'] == 'Northern Africa']['counts'],
    name = 'Northern Africa',
    mode = 'markers',
    marker = dict(
        size = 10,
        #opacity = 0.5,
        color = 'red',
        line = dict(
            width = 2,
        )
    ),
    text = df_pri[df_pri['region'] == 'Northern Africa']['country']
)

trace1 = go.Scatter(
    x= df_pri[df_pri['region'] == 'Western Africa']['priority_idx'],
    y= df_pri[df_pri['region'] == 'Western Africa']['counts'],
    name = 'Western Africa',
    mode = 'markers',
    marker = dict(
        size = 10,
        #opacity = 0.5,
        color = 'indianred',
        line = dict(
            width = 2,
        )
    ),
    text = df_pri[df_pri['region'] == 'Western Africa']['country']
)

trace2 = go.Scatter(
    x= df_pri[df_pri['region'] == 'Eastern Africa']['priority_idx'],
    y= df_pri[df_pri['region'] == 'Eastern Africa']['counts'],
    name = 'Eastern Africa',
    mode = 'markers',
    marker = dict(
        size = 10,
        #opacity = 0.5,
        color = 'lightcoral',
        line = dict(
            width = 2,
        )
    ),
    text = df_pri[df_pri['region'] == 'Eastern Africa']['country']
)

trace3 = go.Scatter(
    x= df_pri[df_pri['region'] == 'Middle Africa']['priority_idx'],
    y= df_pri[df_pri['region'] == 'Middle Africa']['counts'],
    name = 'Middle Africa',
    mode = 'markers',
    marker = dict(
        size = 10,
        #opacity = 0.5,
        color = 'firebrick',
        line = dict(
            width = 2,
        )
    ),
    text = df_pri[df_pri['region'] == 'Middle Africa']['country']
)

trace4 = go.Scatter(
    x= df_pri[df_pri['region'] == 'Southern Africa']['priority_idx'],
    y= df_pri[df_pri['region'] == 'Southern Africa']['counts'],
    name = 'Southern Africa',
    mode = 'markers',
    marker = dict(
        size = 10,
        #opacity = 0.5,
        color = 'crimson',
        line = dict(
            width = 2,
        )
    ),
    text = df_pri[df_pri['region'] == 'Southern Africa']['country']
)

trace5 = go.Scatter(
    x= df_pri[df_pri['region'] == 'Central America']['priority_idx'],
    y= df_pri[df_pri['region'] == 'Central America']['counts'],
    name = 'Central America',
    mode = 'markers',
    marker = dict(
        size = 10,
        #opacity = 0.5,
        color = 'orange',
        line = dict(
            width = 2,
        )
    ),
    text = df_pri[df_pri['region'] == 'Central America']['country']
)

trace6 = go.Scatter(
    x= df_pri[df_pri['region'] == 'South America']['priority_idx'],
    y= df_pri[df_pri['region'] == 'South America']['counts'],
    name = 'South America',
    mode = 'markers',
    marker = dict(
        size = 10,
        #opacity = 0.5,
        color = 'yellow',
        line = dict(
            width = 2,
        )
    ),
    text = df_pri[df_pri['region'] == 'South America']['country']
)

trace7 = go.Scatter(
    x= df_pri[df_pri['region'] == 'Northern Europe']['priority_idx'],
    y= df_pri[df_pri['region'] == 'Northern Europe']['counts'],
    name = 'Northern Europe',
    mode = 'markers',
    marker = dict(
        size = 10,
        #opacity = 0.5,
        color = 'green',
        line = dict(
            width = 2,
        )
    ),
    text = df_pri[df_pri['region'] == 'Northern Europe']['country']
)

trace8 = go.Scatter(
    x= df_pri[df_pri['region'] == 'Eastern Europe']['priority_idx'],
    y= df_pri[df_pri['region'] == 'Eastern Europe']['counts'],
    name = 'Eastern Europe',
    mode = 'markers',
    marker = dict(
        size = 10,
                #opacity = 0.5,
        color = 'lime',
        line = dict(
            width = 2,
        )
    ),
    text = df_pri[df_pri['region'] == 'Eastern Europe']['country']
)

trace9 = go.Scatter(
    x= df_pri[df_pri['region'] == 'Southern Europe']['priority_idx'],
    y= df_pri[df_pri['region'] == 'Southern Europe']['counts'],
    name = 'Southern Europe',
    mode = 'markers',
    marker = dict(
        size = 10,
        #opacity = 0.5,
        color = 'limegreen',
        line = dict(
            width = 2,
        )
    ),
    text = df_pri[df_pri['region'] == 'Southern Europe']['country']
)

trace10 = go.Scatter(
    x= df_pri[df_pri['region'] == 'Western Asia']['priority_idx'],
    y= df_pri[df_pri['region'] == 'Western Asia']['counts'],
    name = 'Western Asia',
    mode = 'markers',
    marker = dict(
        size = 10,
        #opacity = 0.5,
        color = 'blue',
        line = dict(
            width = 2,
        )
    ),
    text = df_pri[df_pri['region'] == 'Western Asia']['country']
)

trace11 = go.Scatter(
    x= df_pri[df_pri['region'] == 'Eastern Asia']['priority_idx'],
    y= df_pri[df_pri['region'] == 'Eastern Asia']['counts'],
    name = 'Eastern Asia',
    mode = 'markers',
    marker = dict(
        size = 10,
        #opacity = 0.5,
        color = 'royalblue',
        line = dict(
            width = 2,
        )
    ),
    text = df_pri[df_pri['region'] == 'Eastern Asia']['country']
)

trace12 = go.Scatter(
    x= df_pri[df_pri['region'] == 'Central Asia']['priority_idx'],
    y= df_pri[df_pri['region'] == 'Central Asia']['counts'],
    name = 'Central Asia',
    mode = 'markers',
    marker = dict(
        size = 10,
        #opacity = 0.5,
        color = 'deepskyblue',
        line = dict(
            width = 2,
        )
    ),
    text = df_pri[df_pri['region'] == 'Central Asia']['country']
)

trace13 = go.Scatter(
    x= df_pri[df_pri['region'] == 'Southern Asia']['priority_idx'],
    y= df_pri[df_pri['region'] == 'Southern Asia']['counts'],
    name = 'Southern Asia',
    mode = 'markers',
    marker = dict(
        size = 10,
        #opacity = 0.5,
        color = 'navy',
        line = dict(
            width = 2,
        )
    ),
    text = df_pri[df_pri['region'] == 'Southern Asia']['country']
)

trace14 = go.Scatter(
    x= df_pri[df_pri['region'] == 'South-eastern Asia']['priority_idx'],
    y= df_pri[df_pri['region'] == 'South-eastern Asia']['counts'],
    name = 'South-eastern Asia',
    mode = 'markers',
    marker = dict(
        size = 10,
        #opacity = 0.5,
        color = 'cornflowerblue',
        line = dict(
            width = 2,
        )
    ),
    text = df_pri[df_pri['region'] == 'South-eastern Asia']['country']
)

trace15 = go.Scatter(
    x= df_pri[df_pri['region'] == 'Caribbean']['priority_idx'],
    y= df_pri[df_pri['region'] == 'Caribbean']['counts'],
    name = 'Caribbean',
    mode = 'markers',
    marker = dict(
        size = 10,
        #opacity = 0.5,
        color = 'magenta',
        line = dict(
            width = 2,
        )
    ),
    text = df_pri[df_pri['region'] == 'Caribbean']['country']
)

data = [trace0, trace1, trace2, trace3, trace4, trace5, trace6, trace7, trace8, trace9, trace10, trace11, trace12, trace13, trace14, trace15]

layout = go.Layout(
    title= 'Countries by Priority Index and Loan Count',
    hovermode= 'closest',
    xaxis= dict(
        title= 'Priority Index (increasing = higher priority)',
        ticklen= 5,
        zeroline= False,
        gridwidth= 2,
    ),
    yaxis=dict(
        title= 'Loan Count',
        ticklen= 5,
        gridwidth= 2,
    ),
    showlegend= True
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
trace0 = go.Scatter(
    x= df_pri[df_pri['region'] == 'Northern Africa']['priority_idx'],
    y= df_pri[df_pri['region'] == 'Northern Africa']['loans_per_cap'],
    name = 'Northern Africa',
    mode = 'markers',
    marker = dict(
        size = 10,
        #opacity = 0.5,
        color = 'red',
        line = dict(
            width = 2,
        )
    ),
    text = df_pri[df_pri['region'] == 'Northern Africa']['country']
)

trace1 = go.Scatter(
    x= df_pri[df_pri['region'] == 'Western Africa']['priority_idx'],
    y= df_pri[df_pri['region'] == 'Western Africa']['loans_per_cap'],
    name = 'Western Africa',
    mode = 'markers',
    marker = dict(
        size = 10,
        #opacity = 0.5,
        color = 'indianred',
        line = dict(
            width = 2,
        )
    ),
    text = df_pri[df_pri['region'] == 'Western Africa']['country']
)

trace2 = go.Scatter(
    x= df_pri[df_pri['region'] == 'Eastern Africa']['priority_idx'],
    y= df_pri[df_pri['region'] == 'Eastern Africa']['loans_per_cap'],
    name = 'Eastern Africa',
    mode = 'markers',
    marker = dict(
        size = 10,
        #opacity = 0.5,
        color = 'lightcoral',
        line = dict(
            width = 2,
        )
    ),
    text = df_pri[df_pri['region'] == 'Eastern Africa']['country']
)

trace3 = go.Scatter(
    x= df_pri[df_pri['region'] == 'Middle Africa']['priority_idx'],
    y= df_pri[df_pri['region'] == 'Middle Africa']['loans_per_cap'],
    name = 'Middle Africa',
    mode = 'markers',
    marker = dict(
        size = 10,
        #opacity = 0.5,
        color = 'firebrick',
        line = dict(
            width = 2,
        )
    ),
    text = df_pri[df_pri['region'] == 'Middle Africa']['country']
)

trace4 = go.Scatter(
    x= df_pri[df_pri['region'] == 'Southern Africa']['priority_idx'],
    y= df_pri[df_pri['region'] == 'Southern Africa']['loans_per_cap'],
    name = 'Southern Africa',
    mode = 'markers',
    marker = dict(
        size = 10,
        #opacity = 0.5,
        color = 'crimson',
        line = dict(
            width = 2,
        )
    ),
    text = df_pri[df_pri['region'] == 'Southern Africa']['country']
)

trace5 = go.Scatter(
    x= df_pri[df_pri['region'] == 'Central America']['priority_idx'],
    y= df_pri[df_pri['region'] == 'Central America']['loans_per_cap'],
    name = 'Central America',
    mode = 'markers',
    marker = dict(
        size = 10,
        #opacity = 0.5,
        color = 'orange',
        line = dict(
            width = 2,
        )
    ),
    text = df_pri[df_pri['region'] == 'Central America']['country']
)

trace6 = go.Scatter(
    x= df_pri[df_pri['region'] == 'South America']['priority_idx'],
    y= df_pri[df_pri['region'] == 'South America']['loans_per_cap'],
    name = 'South America',
    mode = 'markers',
    marker = dict(
        size = 10,
        #opacity = 0.5,
        color = 'yellow',
        line = dict(
            width = 2,
        )
    ),
    text = df_pri[df_pri['region'] == 'South America']['country']
)

trace7 = go.Scatter(
    x= df_pri[df_pri['region'] == 'Northern Europe']['priority_idx'],
    y= df_pri[df_pri['region'] == 'Northern Europe']['loans_per_cap'],
    name = 'Northern Europe',
    mode = 'markers',
    marker = dict(
        size = 10,
        #opacity = 0.5,
        color = 'green',
        line = dict(
            width = 2,
        )
    ),
    text = df_pri[df_pri['region'] == 'Northern Europe']['country']
)

trace8 = go.Scatter(
    x= df_pri[df_pri['region'] == 'Eastern Europe']['priority_idx'],
    y= df_pri[df_pri['region'] == 'Eastern Europe']['loans_per_cap'],
    name = 'Eastern Europe',
    mode = 'markers',
    marker = dict(
        size = 10,
                #opacity = 0.5,
        color = 'lime',
        line = dict(
            width = 2,
        )
    ),
    text = df_pri[df_pri['region'] == 'Eastern Europe']['country']
)

trace9 = go.Scatter(
    x= df_pri[df_pri['region'] == 'Southern Europe']['priority_idx'],
    y= df_pri[df_pri['region'] == 'Southern Europe']['loans_per_cap'],
    name = 'Southern Europe',
    mode = 'markers',
    marker = dict(
        size = 10,
        #opacity = 0.5,
        color = 'limegreen',
        line = dict(
            width = 2,
        )
    ),
    text = df_pri[df_pri['region'] == 'Southern Europe']['country']
)

trace10 = go.Scatter(
    x= df_pri[df_pri['region'] == 'Western Asia']['priority_idx'],
    y= df_pri[df_pri['region'] == 'Western Asia']['loans_per_cap'],
    name = 'Western Asia',
    mode = 'markers',
    marker = dict(
        size = 10,
        #opacity = 0.5,
        color = 'blue',
        line = dict(
            width = 2,
        )
    ),
    text = df_pri[df_pri['region'] == 'Western Asia']['country']
)

trace11 = go.Scatter(
    x= df_pri[df_pri['region'] == 'Eastern Asia']['priority_idx'],
    y= df_pri[df_pri['region'] == 'Eastern Asia']['loans_per_cap'],
    name = 'Eastern Asia',
    mode = 'markers',
    marker = dict(
        size = 10,
        #opacity = 0.5,
        color = 'royalblue',
        line = dict(
            width = 2,
        )
    ),
    text = df_pri[df_pri['region'] == 'Eastern Asia']['country']
)

trace12 = go.Scatter(
    x= df_pri[df_pri['region'] == 'Central Asia']['priority_idx'],
    y= df_pri[df_pri['region'] == 'Central Asia']['loans_per_cap'],
    name = 'Central Asia',
    mode = 'markers',
    marker = dict(
        size = 10,
        #opacity = 0.5,
        color = 'deepskyblue',
        line = dict(
            width = 2,
        )
    ),
    text = df_pri[df_pri['region'] == 'Central Asia']['country']
)

trace13 = go.Scatter(
    x= df_pri[df_pri['region'] == 'Southern Asia']['priority_idx'],
    y= df_pri[df_pri['region'] == 'Southern Asia']['loans_per_cap'],
    name = 'Southern Asia',
    mode = 'markers',
    marker = dict(
        size = 10,
        #opacity = 0.5,
        color = 'navy',
        line = dict(
            width = 2,
        )
    ),
    text = df_pri[df_pri['region'] == 'Southern Asia']['country']
)

trace14 = go.Scatter(
    x= df_pri[df_pri['region'] == 'South-eastern Asia']['priority_idx'],
    y= df_pri[df_pri['region'] == 'South-eastern Asia']['loans_per_cap'],
    name = 'South-eastern Asia',
    mode = 'markers',
    marker = dict(
        size = 10,
        #opacity = 0.5,
        color = 'cornflowerblue',
        line = dict(
            width = 2,
        )
    ),
    text = df_pri[df_pri['region'] == 'South-eastern Asia']['country']
)

trace15 = go.Scatter(
    x= df_pri[df_pri['region'] == 'Caribbean']['priority_idx'],
    y= df_pri[df_pri['region'] == 'Caribbean']['loans_per_cap'],
    name = 'Caribbean',
    mode = 'markers',
    marker = dict(
        size = 10,
        #opacity = 0.5,
        color = 'magenta',
        line = dict(
            width = 2,
        )
    ),
    text = df_pri[df_pri['region'] == 'Caribbean']['country']
)

data = [trace0, trace1, trace2, trace3, trace4, trace5, trace6, trace7, trace8, trace9, trace10, trace11, trace12, trace13, trace14, trace15]

layout = go.Layout(
    title= 'Countries by Priority Index and Loan Count Per Capita',
    hovermode= 'closest',
    xaxis= dict(
        title= 'Priority Index (increasing = higher priority)',
        ticklen= 5,
        zeroline= False,
        gridwidth= 2,
    ),
    yaxis=dict(
        title= 'Loan Count',
        ticklen= 5,
        gridwidth= 2,
    ),
    showlegend= True
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
trace0 = go.Scatter(
    x= df_pri[df_pri['region'] == 'Northern Africa']['priority_idx'],
    y= df_pri[df_pri['region'] == 'Northern Africa']['loans_per_cap'],
    name = 'Northern Africa',
    mode = 'markers',
    marker = dict(
        size = 10,
        #opacity = 0.5,
        color = 'red',
        line = dict(
            width = 2,
        )
    ),
    text = df_pri[df_pri['region'] == 'Northern Africa']['country']
)

trace1 = go.Scatter(
    x= df_pri[df_pri['region'] == 'Western Africa']['priority_idx'],
    y= df_pri[df_pri['region'] == 'Western Africa']['loans_per_cap'],
    name = 'Western Africa',
    mode = 'markers',
    marker = dict(
        size = 10,
        #opacity = 0.5,
        color = 'indianred',
        line = dict(
            width = 2,
        )
    ),
    text = df_pri[df_pri['region'] == 'Western Africa']['country']
)

trace2 = go.Scatter(
    x= df_pri[df_pri['region'] == 'Eastern Africa']['priority_idx'],
    y= df_pri[df_pri['region'] == 'Eastern Africa']['loans_per_cap'],
    name = 'Eastern Africa',
    mode = 'markers',
    marker = dict(
        size = 10,
        #opacity = 0.5,
        color = 'lightcoral',
        line = dict(
            width = 2,
        )
    ),
    text = df_pri[df_pri['region'] == 'Eastern Africa']['country']
)

trace3 = go.Scatter(
    x= df_pri[df_pri['region'] == 'Middle Africa']['priority_idx'],
    y= df_pri[df_pri['region'] == 'Middle Africa']['loans_per_cap'],
    name = 'Middle Africa',
    mode = 'markers',
    marker = dict(
        size = 10,
        #opacity = 0.5,
        color = 'firebrick',
        line = dict(
            width = 2,
        )
    ),
    text = df_pri[df_pri['region'] == 'Middle Africa']['country']
)

trace4 = go.Scatter(
    x= df_pri[df_pri['region'] == 'Southern Africa']['priority_idx'],
    y= df_pri[df_pri['region'] == 'Southern Africa']['loans_per_cap'],
    name = 'Southern Africa',
    mode = 'markers',
    marker = dict(
        size = 10,
        #opacity = 0.5,
        color = 'crimson',
        line = dict(
            width = 2,
        )
    ),
    text = df_pri[df_pri['region'] == 'Southern Africa']['country']
)

trace5 = go.Scatter(
    x= df_pri[df_pri['region'] == 'Central America']['priority_idx'],
    y= df_pri[df_pri['region'] == 'Central America']['loans_per_cap'],
    name = 'Central America',
    mode = 'markers',
    marker = dict(
        size = 10,
        #opacity = 0.5,
        color = 'orange',
        line = dict(
            width = 2,
        )
    ),
    text = df_pri[df_pri['region'] == 'Central America']['country']
)

trace6 = go.Scatter(
    x= df_pri[df_pri['region'] == 'South America']['priority_idx'],
    y= df_pri[df_pri['region'] == 'South America']['loans_per_cap'],
    name = 'South America',
    mode = 'markers',
    marker = dict(
        size = 10,
        #opacity = 0.5,
        color = 'yellow',
        line = dict(
            width = 2,
        )
    ),
    text = df_pri[df_pri['region'] == 'South America']['country']
)

trace7 = go.Scatter(
    x= df_pri[df_pri['region'] == 'Northern Europe']['priority_idx'],
    y= df_pri[df_pri['region'] == 'Northern Europe']['loans_per_cap'],
    name = 'Northern Europe',
    mode = 'markers',
    marker = dict(
        size = 10,
        #opacity = 0.5,
        color = 'green',
        line = dict(
            width = 2,
        )
    ),
    text = df_pri[df_pri['region'] == 'Northern Europe']['country']
)

trace8 = go.Scatter(
    x= df_pri[df_pri['region'] == 'Eastern Europe']['priority_idx'],
    y= df_pri[df_pri['region'] == 'Eastern Europe']['loans_per_cap'],
    name = 'Eastern Europe',
    mode = 'markers',
    marker = dict(
        size = 10,
                #opacity = 0.5,
        color = 'lime',
        line = dict(
            width = 2,
        )
    ),
    text = df_pri[df_pri['region'] == 'Eastern Europe']['country']
)

trace9 = go.Scatter(
    x= df_pri[df_pri['region'] == 'Southern Europe']['priority_idx'],
    y= df_pri[df_pri['region'] == 'Southern Europe']['loans_per_cap'],
    name = 'Southern Europe',
    mode = 'markers',
    marker = dict(
        size = 10,
        #opacity = 0.5,
        color = 'limegreen',
        line = dict(
            width = 2,
        )
    ),
    text = df_pri[df_pri['region'] == 'Southern Europe']['country']
)

trace10 = go.Scatter(
    x= df_pri[df_pri['region'] == 'Western Asia']['priority_idx'],
    y= df_pri[df_pri['region'] == 'Western Asia']['loans_per_cap'],
    name = 'Western Asia',
    mode = 'markers',
    marker = dict(
        size = 10,
        #opacity = 0.5,
        color = 'blue',
        line = dict(
            width = 2,
        )
    ),
    text = df_pri[df_pri['region'] == 'Western Asia']['country']
)

trace11 = go.Scatter(
    x= df_pri[df_pri['region'] == 'Eastern Asia']['priority_idx'],
    y= df_pri[df_pri['region'] == 'Eastern Asia']['loans_per_cap'],
    name = 'Eastern Asia',
    mode = 'markers',
    marker = dict(
        size = 10,
        #opacity = 0.5,
        color = 'royalblue',
        line = dict(
            width = 2,
        )
    ),
    text = df_pri[df_pri['region'] == 'Eastern Asia']['country']
)

trace12 = go.Scatter(
    x= df_pri[df_pri['region'] == 'Central Asia']['priority_idx'],
    y= df_pri[df_pri['region'] == 'Central Asia']['loans_per_cap'],
    name = 'Central Asia',
    mode = 'markers',
    marker = dict(
        size = 10,
        #opacity = 0.5,
        color = 'deepskyblue',
        line = dict(
            width = 2,
        )
    ),
    text = df_pri[df_pri['region'] == 'Central Asia']['country']
)

trace13 = go.Scatter(
    x= df_pri[df_pri['region'] == 'Southern Asia']['priority_idx'],
    y= df_pri[df_pri['region'] == 'Southern Asia']['loans_per_cap'],
    name = 'Southern Asia',
    mode = 'markers',
    marker = dict(
        size = 10,
        #opacity = 0.5,
        color = 'navy',
        line = dict(
            width = 2,
        )
    ),
    text = df_pri[df_pri['region'] == 'Southern Asia']['country']
)

trace14 = go.Scatter(
    x= df_pri[df_pri['region'] == 'South-eastern Asia']['priority_idx'],
    y= df_pri[df_pri['region'] == 'South-eastern Asia']['loans_per_cap'],
    name = 'South-eastern Asia',
    mode = 'markers',
    marker = dict(
        size = 10,
        #opacity = 0.5,
        color = 'cornflowerblue',
        line = dict(
            width = 2,
        )
    ),
    text = df_pri[df_pri['region'] == 'South-eastern Asia']['country']
)

trace15 = go.Scatter(
    x= df_pri[df_pri['region'] == 'Caribbean']['priority_idx'],
    y= df_pri[df_pri['region'] == 'Caribbean']['loans_per_cap'],
    name = 'Caribbean',
    mode = 'markers',
    marker = dict(
        size = 10,
        #opacity = 0.5,
        color = 'magenta',
        line = dict(
            width = 2,
        )
    ),
    text = df_pri[df_pri['region'] == 'Caribbean']['country']
)

data = [trace0, trace1, trace2, trace3, trace4, trace5, trace6, trace7, trace8, trace9, trace10, trace11, trace12, trace13, trace14, trace15]

layout = go.Layout(
    title= 'Countries by Priority Index and Loan Count Per Capita < 0.002',
    hovermode= 'closest',
    xaxis= dict(
        title= 'Priority Index (increasing = higher priority)',
        ticklen= 5,
        zeroline= False,
        gridwidth= 2,
    ),
    yaxis=dict(
        title= 'Loan Count',
        ticklen= 5,
        gridwidth= 2,
        range=[0, 0.002]
    ),
    showlegend= True
)
    
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
df_kv_loans['loan_URL'] = df_kv_loans['id'].apply(lambda x: 'https://www.kiva.org/lend/' + str(x))
HTML('<iframe width="560" height="315" src="https://www.youtube.com/embed/FwAj3qTPt4E?rel=0&amp;controls=0&amp;showinfo=0" frameborder="0" allowfullscreen></iframe>')
df_kv_loans[(df_kv_loans['use'].fillna('').str.contains('toilet')) 
    & ((df_kv_loans['country'] == 'India') | (df_kv_loans['country'] == 'Indonesia'))
    & ((df_kv_loans['sector'] == 'Housing') | (df_kv_loans['sector'] == 'Health'))
    ].head(10)[['id', 'funded_amount', 'sector', 'activity', 'country', 'use', 'partner_id', 'loan_URL']]
df_kv_loans[(df_kv_loans['use'] == 'to buy a sound system for her house.')][['id', 'funded_amount', 'sector', 'activity', 'country', 'use', 'partner_id', 'loan_URL', 'lender_count']]
