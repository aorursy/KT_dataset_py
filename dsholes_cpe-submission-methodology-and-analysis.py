import numpy as np
import geopandas as gpd
import pandas as pd

from matplotlib import pyplot as plt
import plotly.offline as plotly
import plotly.graph_objs as go
import plotly.tools as tools

plotly.init_notebook_mode(connected=True)

%matplotlib inline
police_uof_path = "../input/data-science-for-good/cpe-data/Dept_37-00027/37-00027_UOF-P_2014-2016_prepped.csv"

# Use Pandas to read the "prepped" CSV, dropping the first row, which is just more headers
police_uof_df = pd.read_csv(police_uof_path).iloc[1:].reset_index(drop=True)
police_shp_path = "../input/data-science-for-good/cpe-data/Dept_37-00027/37-00027_Shapefiles/APD_DIST.shp"

# Use Geopandas to read the Shapefile
police_shp_gdf = gpd.read_file(police_shp_path)
police_shp_gdf.crs = {'init' :'esri:102739'}
police_shp_gdf = police_shp_gdf.to_crs(epsg='4326')
census_shp_path = "../input/cb-2017-48-tract-500k/cb_2017_48_tract_500k.shp"
census_tracts_gdf = gpd.read_file(census_shp_path)
census_tracts_gdf['GEOID'] = census_tracts_gdf['GEOID'].astype('int64')
census_tracts_gdf = census_tracts_gdf.to_crs(epsg='4326')
census_race_file_path = "../input/texas-14-5yr-dp05-racesexage/ACS_14_5YR_DP05.csv"


census_race_meta_file_path = "../input/texas-14-5yr-dp05-racesexage/ACS_14_5YR_DP05_metadata.csv"

census_race_df = pd.read_csv(census_race_file_path)
census_race_meta_df = pd.read_csv(census_race_meta_file_path,skiprows=2)
census_race_meta_df.columns = ['header','description']
meta_mask = ~census_race_meta_df['description'].str.contains('Margin of Error|SEX AND AGE',regex=True)
census_race_meta_df = census_race_meta_df[meta_mask] # Only keep "Percent" categories

census_columns_to_drop = ['HC02','HC04']
column_mask = census_race_df.columns.str.contains('|'.join(census_columns_to_drop), regex=True)
census_race_df = census_race_df.loc[:,census_race_df.columns[~column_mask]]
census_race_df = census_race_df.iloc[1:].reset_index(drop=True)

# Rename Census Tract ID column in ACS Poverty CSV to align with Census Tract Shapefile
census_race_df = census_race_df.rename(columns={'GEO.id2':'GEOID'})
print('Police UOF Race Labels:')
print(police_uof_df.SUBJECT_RACE.value_counts(dropna=False))
police_uof_df = police_uof_df.dropna(subset=['SUBJECT_RACE']).reset_index(drop=True)
race_perc_col_begin = 'Percent; HISPANIC OR LATINO AND RACE - Total population - '
race_est_col_begin = 'Estimate; HISPANIC OR LATINO AND RACE - Total population - '
race_col_end = ['Hispanic or Latino (of any race)',
                'Not Hispanic or Latino - White alone',
                'Not Hispanic or Latino - Black or African American alone',
                'Not Hispanic or Latino - American Indian and Alaska Native alone',
                'Not Hispanic or Latino - Asian alone',
                'Not Hispanic or Latino - Native Hawaiian and Other Pacific Islander alone',
                'Not Hispanic or Latino - Some other race alone',
                'Not Hispanic or Latino - Two or more races']
race_perc_cols = race_perc_col_begin+np.array(race_col_end,dtype=object)
race_est_cols = race_est_col_begin+np.array(race_col_end,dtype=object)
census_race_meta_df[census_race_meta_df['description'].isin(race_est_cols)].header.values
meta_perc_mask = census_race_meta_df['description'].isin(race_perc_cols)
meta_est_mask = census_race_meta_df['description'].isin(race_est_cols)
race_perc_headers = census_race_meta_df[meta_perc_mask].header.values
race_est_headers = census_race_meta_df[meta_est_mask].header.values
race_desc = np.array(['Hispanic','White','Black','Native American','Asian',
                      'Pacific Islander','Some Other Race','Two or more races']).astype(object)
race_perc_header_to_desc = dict(zip(race_perc_headers,race_desc+', Percent'))
race_est_header_to_desc = dict(zip(race_est_headers,race_desc+', Estimate'))
for item in list(zip(race_est_headers,race_desc)):
    print('{0}: {1}, Estimate'.format(item[0],item[1]))
print('')
for item in list(zip(race_perc_headers,race_desc)):
    print('{0}: {1}, Percent'.format(item[0],item[1]))
print(census_race_df.loc[0,race_perc_headers])
print('')
print('Sum: {0}'.format(census_race_df.loc[0,race_perc_headers].sum()))
AX_LABEL_FONT_DICT = {'size':14}
AX_TITLE_FONT_DICT = {'size':16}

fig0,ax0 = plt.subplots()
(police_shp_gdf.dissolve(by='SECTOR').reset_index()
 .plot(ax=ax0,column='SECTOR',legend=True))
ax0.set_xlabel('Latitude (deg)',fontdict=AX_LABEL_FONT_DICT)
ax0.set_ylabel('Longitude (deg)',fontdict=AX_LABEL_FONT_DICT)
ax0.set_title('Dept_37-00027: Police Precincts (SECTORS)',
              fontdict=AX_TITLE_FONT_DICT)
leg0 = ax0.get_legend()
leg0.set_bbox_to_anchor((1.28, 1., 0., 0.))
leg0.set_title('SECTOR',prop={'size':12})
fig0.set_size_inches(7,7)
fig1,ax1 = plt.subplots()
census_tracts_gdf.plot(ax=ax1,color='#74b9ff',alpha=.4,edgecolor='white')
police_shp_gdf.plot(ax=ax1,column='SECTOR')
ax1.set_xlabel('Latitude (deg)',fontdict=AX_LABEL_FONT_DICT)
ax1.set_ylabel('Longitude (deg)',fontdict=AX_LABEL_FONT_DICT)
ax1.set_title('Dept_37-00027 and Texas Census Tracts',
              fontdict=AX_TITLE_FONT_DICT)
fig1.set_size_inches(8,11)
police_sector_shp_gdf = police_shp_gdf.dissolve(by='SECTOR').reset_index()
#joined_df = gpd.overlay(police_sector_shp_gdf,census_tracts_gdf)
joined_df = pd.read_pickle("../input/cpe-joined-df/cpe_joined_df.pkl")
# fig2,ax2 = plt.subplots()
# joined_df.plot(ax=ax2,column='SECTOR',legend=True)
# ax2.set_xlabel('Latitude (deg)',fontdict=AX_LABEL_FONT_DICT)
# ax2.set_ylabel('Longitude (deg)',fontdict=AX_LABEL_FONT_DICT)
# ax2.set_title('Dept_37-00027: Police Precincts with Overlayed Census Tracts',
#               fontdict=AX_TITLE_FONT_DICT)
# leg2 = ax2.get_legend()
# leg2.set_bbox_to_anchor((1.28, 1., 0., 0.))
# leg2.set_title('SECTOR',prop={'size':12})
# fig2.set_size_inches(7,7)
def perc_tract_area(group, tracts_gdf):
    joined_area = group.area.values
    orig_tract_area = tracts_gdf.set_index('GEOID').loc[group['GEOID'].values,:].area.values
    perc_of_orig_tract = joined_area/orig_tract_area
    group['perc_of_orig_tract'] = perc_of_orig_tract
    return group
print(census_race_meta_df[census_race_meta_df['header'] == 'HC01_VC43'])
est_tot_pop_header = 'HC01_VC43'
joined_with_perc_area = (joined_df
                         .groupby('SECTOR')
                         .apply(perc_tract_area, census_tracts_gdf)
                         .sort_index())

joined_perc_area_and_pop = (joined_with_perc_area
                            .merge(census_race_df[['GEOID',est_tot_pop_header]],
                                   on='GEOID')
                            .sort_values('SECTOR')
                            .reset_index())

# Adjusting population based on percent area of census tract within police district
joined_perc_area_and_pop['pop_adj_by_area'] = (joined_perc_area_and_pop['perc_of_orig_tract']*
                                               joined_perc_area_and_pop[est_tot_pop_header])
def adj_pop_weight_factor(group):
    group['weight_factor'] = group['pop_adj_by_area']/group['pop_adj_by_area'].sum()
    return group
# Calculate areal_fractions to use as weight factors
joined_pop_weight_factor = (joined_perc_area_and_pop
                            .groupby('SECTOR')
                            .apply(adj_pop_weight_factor)
                            .drop('index',axis=1))

race_est_merge_headers = np.insert(race_est_headers,0,'GEOID')


joined_est_pop_weighted = (joined_pop_weight_factor
                           .merge(census_race_df[race_est_merge_headers],on='GEOID')
                           .sort_values('SECTOR').reset_index(drop=True))

# Use areal_fractions to "re-distribute" population of partial census tracts
est_pop_weighted = (joined_est_pop_weighted[race_est_headers]
                    .multiply(joined_est_pop_weighted['weight_factor'], 
                              axis="index"))


joined_est_pop_weighted = (pd.concat([joined_est_pop_weighted,
                                      est_pop_weighted.rename(columns=race_est_header_to_desc)],
                                      axis=1)
                           .drop(race_est_headers,axis=1))

# Sum all intersected_tract populations
race_est_by_pol_sector = (joined_est_pop_weighted
                          .groupby('SECTOR')[list(race_est_header_to_desc.values())]
                          .sum())
race_perc_by_pol_sector = (race_est_by_pol_sector
                           .divide(race_est_by_pol_sector.sum(axis=1),
                                   axis='index'))*100.
race_perc_by_pol_sector.columns = (race_perc_by_pol_sector
                                   .columns
                                   .str
                                   .replace('Estimate','Percent'))
race_perc_by_pol_sector.sort_index(axis=1,inplace=True)
race_perc_by_pol_sector
sectors_in_pol_shp = police_shp_gdf.SECTOR.value_counts(dropna=False).sort_index()
print(sectors_in_pol_shp)
print(police_uof_df.LOCATION_DISTRICT.value_counts().sort_index())
mask_missing_sectors = ~police_uof_df.LOCATION_DISTRICT.isin(['-','88',np.nan])
police_uof_df = police_uof_df[mask_missing_sectors].reset_index(drop=True)
sectors_in_pol_uof = police_uof_df.LOCATION_DISTRICT.value_counts().sort_index()
print(sectors_in_pol_uof)
sector_abbrev_dict = dict(zip(sectors_in_pol_uof.index,
                              sectors_in_pol_shp.index))
sector_abbrev_dict
police_uof_df.LOCATION_DISTRICT.replace(sector_abbrev_dict,inplace=True)
race_uof_est_by_pol_sector = (police_uof_df
                              .groupby('LOCATION_DISTRICT')['SUBJECT_RACE']
                              .value_counts()
                              .unstack())
race_uof_est_by_pol_sector.fillna(0,inplace=True)
race_uof_est_by_pol_sector.columns = race_uof_est_by_pol_sector.columns+', Estimate'
race_uof_est_by_pol_sector
race_uof_perc_by_pol_sector = (race_uof_est_by_pol_sector
                               .divide(race_uof_est_by_pol_sector.sum(axis=1),
                                       axis='index'))*100.
race_uof_perc_by_pol_sector.columns = (race_uof_perc_by_pol_sector
                                       .columns
                                       .str
                                       .replace('Estimate','Percent'))
race_uof_perc_by_pol_sector.drop('Unknown, Percent',axis=1,inplace=True)
race_uof_perc_by_pol_sector
missing_cols_mask = ~(race_perc_by_pol_sector
                      .columns
                      .isin(race_uof_perc_by_pol_sector.columns))
missing_cols = race_perc_by_pol_sector.columns[missing_cols_mask]
race_uof_perc_by_pol_sector = pd.concat([race_uof_perc_by_pol_sector,
                                         pd.DataFrame(columns=missing_cols)],
                                         axis=1,sort=False).fillna(0)
race_uof_perc_by_pol_sector.sort_index(axis=1,inplace=True)
race_tick_labels = list(race_perc_by_pol_sector.columns.str.replace(', Percent',""))

for district_str in race_perc_by_pol_sector.index:
    race_breakdown_census = race_perc_by_pol_sector.loc[district_str,:]
    trace1 = go.Bar(
        y= race_breakdown_census.values,
        x= race_tick_labels,
        marker=dict(
            color='#34495e',
            line=dict(
                color='rgba(255, 255, 255, 0.0)',
                width=1),
        ),
        name='Census',
        orientation='v',
        showlegend = True
    )

    race_breakdown_uof = race_uof_perc_by_pol_sector.loc[district_str,:]
    trace2 = go.Bar(
        y= race_breakdown_uof.values,
        x= race_tick_labels,
        marker=dict(
            color='#ffa500',
            line=dict(
                color='rgba(255, 255, 255, 0.0)',
                width=1),
        ),
        name='Use of Force',
        orientation='v',
        showlegend = True
    )

    data = [trace1, trace2]
    layout = go.Layout(
        barmode='group',
        title = "Dept_37-00027: Racial Breakdown for Police Precinct (SECTOR) '{0}'".format(district_str)
    )

    fig = go.Figure(data=data, layout=layout)
    plotly.iplot(fig)
