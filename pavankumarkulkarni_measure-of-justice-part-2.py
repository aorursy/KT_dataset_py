#Load libraries needed for the solution
%matplotlib inline
import geopandas as gpd
from shapely.geometry import Point
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
def metadata_summary(file_path):
    ''' Function to summarise metadata file. The file has components of meta data seperated by _ in code and ; in description.
    It returns unique combinations of code and description part. This will help in grouping and extracting only needed columns
    from data.
    Input - Metadata file path.
    Output - Dataframe with code, desc and type unique combinations
    '''
    meta_df = pd.read_csv(file_path,names = ['code','desc'])
    meta_code = meta_df['code'].str.split('_',expand = True)
    meta_desc = meta_df['desc'].str.split(';',expand = True)
    temp_df = pd.DataFrame(columns = ['code','desc','type'])
    for cols in meta_code.columns:
        t = pd.concat([meta_code[cols], meta_desc[cols]], axis=1)
        t['type'] = 'type'+str(cols)
        t.columns = ['code','desc','type']
        temp_df = pd.concat([temp_df,t],axis=0)

    temp_df.drop_duplicates(inplace = True)
    temp_df.dropna(subset = ['code','desc'],inplace = True)
    temp_df.reset_index(inplace = True, drop = True)
    return(temp_df)
def cleveland_plot(xmins, xmaxs, ylabs, chart_title, xmin_lab, xmax_lab,xtitle, ytitle, fsize=(10,10)):
    ''' Plot a cleveland plot showing the xmin and xmax for a set of categorical variable'''
    tot_recs = 10 if len(xmins) > 30 else len(xmins)
    y_range = np.arange(1,tot_recs+1)

    fig, ax = plt.subplots(figsize=fsize)
    plt.hlines(y = y_range, xmin = xmins[:tot_recs], xmax = xmaxs[:tot_recs], 
               alpha=0.3, color='gray', linewidth = 2)
    plt.scatter(x = xmins[:tot_recs],  y = y_range, color = 'green', label = xmin_lab, s =50, marker='o')
    plt.scatter(x = xmaxs[:tot_recs],  y = y_range, color = 'red', label = xmax_lab, s = 50,marker = 'H')
    plt.yticks(y_range, ylabs[:tot_recs],fontsize=12)
    plt.ylabel(ytitle, fontsize = 14, color = 'gray')
    plt.ylim(tot_recs+1,0)
    plt.xticks(fontsize=12)
    plt.xlabel(xtitle, fontsize = 14, color = 'gray')
    plt.legend(fontsize = 12,loc='best')
    plt.title(chart_title, fontsize = 18)
    plt.grid(False)
    ax.set_facecolor('white')
    fig.set_facecolor('w')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
base_path = '../input/data-science-for-good/cpe-data/'
police_records_mn = pd.read_csv(base_path+'Dept_24-00098/24-00098_Vehicle-Stops-data.csv',
                            skiprows=[1])
police_records_mn.shape
police_records_mn.head()
# Missing long/lat will error out. Hence these are dropped.
# create long/lat point, set correct CRS it represented in then convert to common CRS used in this solution.
police_records_mn.dropna(subset = ['LOCATION_LONGITUDE','LOCATION_LATITUDE'],inplace = True)
police_records_mn['geometry'] = list(zip(police_records_mn['LOCATION_LONGITUDE'],police_records_mn['LOCATION_LATITUDE']))
police_records_mn['geometry'] = police_records_mn['geometry'].apply(Point)
police_records_mn = gpd.GeoDataFrame(police_records_mn,
                                 geometry = 'geometry',
                                 crs ={'init': 'epsg:4326'} )
police_records_mn.info()
police_records_mn.crs
police_records_mn.head()
# to get total count by grid num
pr_summary_mn = pd.DataFrame(police_records_mn['LOCATION_DISTRICT'].value_counts()).reset_index()
pr_summary_mn.columns = ['gridnum', 'total_stops']
# get totals by gender of stoppages
t = pd.DataFrame(police_records_mn.groupby(by = ['LOCATION_DISTRICT','SUBJECT_GENDER']).size()).reset_index()
t.columns = ['gridnum','gender','stops']
t = t.pivot_table(values = 'stops',index = 'gridnum', columns = 'gender', fill_value = 0).reset_index()
t.rename(columns = {'No Data':'gender_missing', 'Female' : 'female_stops', 'Male':'male_stops'},inplace = True)
pr_summary_mn = pr_summary_mn.merge(t, how = 'left', on = 'gridnum')
# get totals by race of drivers
t = pd.DataFrame(police_records_mn.groupby(by = ['LOCATION_DISTRICT','SUBJECT_RACE']).size()).reset_index()
t.columns = ['gridnum','race','stops']
t['race'] = t['race'].str.lower()
t['race'] = t['race'].str.replace(' ','_')
t = t.pivot_table(values = 'stops',index = 'gridnum', columns = 'race', fill_value = 0).reset_index()
t['native_american'] = t['native_am'] + t['native_american']
t['other'] = t['other'] + t['no_data']
t.drop(columns = ['native_am','no_data'],inplace = True)
pr_summary_mn = pr_summary_mn.merge(t, how = 'left', on = 'gridnum')
# Calculate percentage columns
pr_summary_mn.sort_values(by = 'gridnum',inplace=True)
pr_summary_mn.reset_index(inplace = True,drop=True)
pr_summary_mn['female_stop_per'] = pr_summary_mn['female_stops']/(pr_summary_mn['female_stops'] + pr_summary_mn['male_stops'])*100
pr_summary_mn['male_stop_per'] = pr_summary_mn['male_stops']/(pr_summary_mn['female_stops'] + pr_summary_mn['male_stops'])*100
pr_summary_mn['asian_stop_per'] = pr_summary_mn['asian']/pr_summary_mn['total_stops']*100
pr_summary_mn['black_stop_per'] = pr_summary_mn['black']/pr_summary_mn['total_stops']*100
pr_summary_mn['latino_stop_per'] = pr_summary_mn['latino']/pr_summary_mn['total_stops']*100
pr_summary_mn['native_american_stop_per'] = pr_summary_mn['native_american']/pr_summary_mn['total_stops']*100
pr_summary_mn['other_stop_per'] = pr_summary_mn['other']/pr_summary_mn['total_stops']*100
pr_summary_mn['white_stop_per'] = pr_summary_mn['white']/pr_summary_mn['total_stops']*100
pr_summary_mn['gridnum'] = pr_summary_mn['gridnum'].astype(int)
pr_summary_mn.head(3)
police_records_mn.INCIDENT_REASON.unique()
stop_code = {'No Data':'no_reason_stop',
             'Investigative Stop':'no_reason_stop',
             'Moving Violation':'with_reason_stop', 
             'Equipment Violation':'with_reason_stop',
             '911 Call / Citizen Reported':'with_reason_stop'}
police_records_mn['stop_category'] = police_records_mn['INCIDENT_REASON'].map(stop_code)
temp = pd.crosstab(police_records_mn['LOCATION_DISTRICT'],police_records_mn['stop_category'],normalize = 'index').reset_index()
temp['LOCATION_DISTRICT'] = temp['LOCATION_DISTRICT'].astype(int)
temp['no_reason_stop'] = temp['no_reason_stop']*100
pr_summary_mn = pr_summary_mn.merge(temp[['LOCATION_DISTRICT','no_reason_stop']], how = 'left', left_on = 'gridnum', right_on = 'LOCATION_DISTRICT')
pr_summary_mn.drop(columns = ['LOCATION_DISTRICT'],inplace = True)
pr_summary_mn.head()
pp_mn_geo = gpd.read_file(base_path + 'Dept_24-00098/24-00098_Shapefiles/StPaul_geo_export_6646246d-0f26-48c5-a924-f5a99bb51c47.shp')
pp_mn_geo.crs
pp_mn_geo.describe()
pp_mn_geo.head(3)
ax = police_records_mn.plot(figsize=(10,10))
pp_mn_geo.plot(ax=ax,facecolor = 'cyan',edgecolor = 'black', alpha=0.2)
ax.set_title('MN police precinct and vehicle stop scatter plot', fontsize=18)
ax.set_axis_off()
pp_mn_geo['center'] = pp_mn_geo['geometry'].centroid
pp_mn_geo['gridnum'] = pp_mn_geo['gridnum'].astype(int)
pp_pr_summary_mn = pd.merge(pp_mn_geo, pr_summary_mn, on = 'gridnum', how = 'inner')
ax = pp_pr_summary_mn.plot(facecolor = 'None', edgecolor = 'grey',figsize = (16,16))
plt.scatter(x=pp_pr_summary_mn['center'].apply(lambda x: Point(x).x),
           y=pp_pr_summary_mn['center'].apply(lambda x: Point(x).y),
           s = (pp_pr_summary_mn['total_stops']/8),
           alpha=0.4,
           c='red')
ax.set_title('Fig - 4 Vehicle stops count in each police precinct', fontsize =18)
for i, pp in enumerate(pp_pr_summary_mn.gridnum):
    ax.annotate(pp,(pp_pr_summary_mn.geometry[i].centroid.x, pp_pr_summary_mn.geometry[i].centroid.y))
ax.set_axis_off()
def choropleth_map(df, c, fsize = (6,6)):
    ax = df.plot(column = c, legend=True, figsize=fsize, cmap = 'Oranges', scheme = 'quantiles')
    ax.set_title('Map - ' +c, fontsize=16)
    ax.get_legend().set_bbox_to_anchor((1.4,1.2))
    ax.set_axis_off()
# plot the choropleth from per columns.
for cl in pp_pr_summary_mn.iloc[:,14:].columns:
    choropleth_map(pp_pr_summary_mn, cl)
def metadata_summary(file_path):
    ''' Function to summarise metadata file. The file has components of meta data seperated by _ in code and ; in description.
    It returns unique combinations of code and description part. This will help in grouping and extracting only needed columns
    from data.
    Input - Metadata file path.
    Output - Dataframe with code, desc and type unique combinations
    '''
    meta_df = pd.read_csv(file_path,names = ['code','desc'])
    meta_code = meta_df['code'].str.split('_',expand = True)
    meta_desc = meta_df['desc'].str.split(';',expand = True)
    temp_df = pd.DataFrame(columns = ['code','desc','type'])
    for cols in meta_code.columns:
        t = pd.concat([meta_code[cols], meta_desc[cols]], axis=1)
        t['type'] = 'type'+str(cols)
        t.columns = ['code','desc','type']
        temp_df = pd.concat([temp_df,t],axis=0)

    temp_df.drop_duplicates(inplace = True)
    temp_df.dropna(subset = ['code','desc'],inplace = True)
    temp_df.reset_index(inplace = True, drop = True)
    return(temp_df)
rsa_mn_path = base_path + 'Dept_24-00098/24-00098_ACS_data/24-00098_ACS_race-sex-age/ACS_15_5YR_DP05_metadata.csv'
rsa_meta_mn = metadata_summary(rsa_mn_path)
rsa_meta_mn.head()
rsa_meta_mn[rsa_meta_mn.desc.str.contains('RACE - One race')]
# Read race_sex_age data
rsa_mn_data = pd.read_csv(base_path + 'Dept_24-00098/24-00098_ACS_data/24-00098_ACS_race-sex-age/ACS_15_5YR_DP05_with_ann.csv',
           skiprows = [1], na_values = '-')
rsa_mn_data.head(3)
acs_mn_data = rsa_mn_data.filter(items = ['GEO.id','HC01_VC03','HC01_VC04','HC01_VC05','HC01_VC49','HC01_VC50','HC01_VC56'])
acs_mn_data.rename(columns = {'HC01_VC03':'tot_pop',
                              'HC01_VC04':'male_pop',
                              'HC01_VC05':'female_pop',
                              'HC01_VC49':'race_white',
                              'HC01_VC50':'race_black',
                              'HC01_VC56':'race_asian'}, inplace = True)
ea_mn_path = base_path + 'Dept_24-00098/24-00098_ACS_data/24-00098_ACS_education-attainment/ACS_15_5YR_S1501_metadata.csv'
ea_meta_mn = metadata_summary(ea_mn_path)
ea_meta_mn.head()
# Read education_attainment
ea_mn_data = pd.read_csv(base_path + 'Dept_24-00098/24-00098_ACS_data/24-00098_ACS_education-attainment/ACS_15_5YR_S1501_with_ann.csv',
           skiprows = [1], na_values = '-')
ea_mn_data = ea_mn_data.filter(items = ['GEO.id','HC01_EST_VC02', 'HC01_EST_VC08','HC01_EST_VC03' ,'HC01_EST_VC09', 'HC01_EST_VC10'])
ea_mn_data['edu_pop_tot'] = ea_mn_data['HC01_EST_VC02'] + ea_mn_data['HC01_EST_VC08']
ea_mn_data['edu_pop_less'] = ea_mn_data['HC01_EST_VC03'] + ea_mn_data['HC01_EST_VC09'] + ea_mn_data['HC01_EST_VC10']
ea_mn_data = ea_mn_data.filter(items = ['GEO.id', 'edu_pop_tot','edu_pop_less'])
acs_mn_data = acs_mn_data.merge(ea_mn_data, on = 'GEO.id', how = 'left')
acs_mn_data.head()
p_mn_path = base_path + 'Dept_24-00098/24-00098_ACS_data/24-00098_ACS_poverty/ACS_15_5YR_S1701_metadata.csv'
p_meta_mn = metadata_summary(p_mn_path)
p_meta_mn.head()
# Poverty data
p_mn_data = pd.read_csv(base_path + 'Dept_24-00098/24-00098_ACS_data/24-00098_ACS_poverty/ACS_15_5YR_S1701_with_ann.csv',
           skiprows = [1], na_values = '-')
p_mn_data = p_mn_data.filter(items = ['GEO.id','HC01_EST_VC01','HC02_EST_VC01'])
p_mn_data.rename(columns = {'HC01_EST_VC01':'pov_pop_tot','HC02_EST_VC01':'pov_pop_below'},inplace = True)
acs_mn_data = acs_mn_data.merge(p_mn_data, on = 'GEO.id', how = 'left')
acs_mn_data.head(3)
emp_mn_path = base_path + 'Dept_24-00098/24-00098_ACS_data/24-00098_ACS_employment/ACS_15_5YR_S2301_metadata.csv'
emp_meta_mn = metadata_summary(emp_mn_path)
emp_meta_mn.head()
# Employment data
emp_mn_data = pd.read_csv(base_path + 'Dept_24-00098/24-00098_ACS_data/24-00098_ACS_employment/ACS_15_5YR_S2301_with_ann.csv',
           skiprows = [1], na_values = '-')
emp_mn_data = emp_mn_data.filter(items = ['GEO.id','HC01_EST_VC01','HC04_EST_VC01'])
emp_mn_data.rename(columns = {'HC01_EST_VC01':'pop_emp_tot','HC04_EST_VC01':'pop_emp_unemploy_rate'},inplace = True)
acs_mn_data = acs_mn_data.merge(emp_mn_data, on = 'GEO.id', how = 'left')
ooh_mn_path = base_path + 'Dept_24-00098/24-00098_ACS_data/24-00098_ACS_owner-occupied-housing/ACS_15_5YR_S2502_metadata.csv'
ooh_meta_mn = metadata_summary(ooh_mn_path)
ooh_meta_mn.head()
# Employment data
ooh_mn_data = pd.read_csv(base_path + 'Dept_24-00098/24-00098_ACS_data/24-00098_ACS_owner-occupied-housing/ACS_15_5YR_S2502_with_ann.csv',
           skiprows = [1], na_values = '-')
ooh_mn_data = ooh_mn_data.filter(items = ['GEO.id','HC01_EST_VC01','HC02_EST_VC01'])
ooh_mn_data.rename(columns = {'HC01_EST_VC01':'occupied_units','HC02_EST_VC01':'owner_occupied_units'},inplace = True)
acs_mn_data = acs_mn_data.merge(ooh_mn_data, on = 'GEO.id', how = 'left')
acs_mn_data['pop_emp_unemployed_count'] = acs_mn_data['pop_emp_tot'] * acs_mn_data['pop_emp_unemploy_rate']/100
acs_mn_data.drop(columns = ['pop_emp_unemploy_rate'], inplace = True)
acs_mn_data.head(3)
# read the census track data and inspect first few rows
#base_path = '../input/cpe-data/' for kaggle
census_tracks_mn = gpd.read_file('../input/mncensustracks/cb_2017_27_tract_500k.shp')
census_tracks_mn.head(2)
census_tracks_mn.crs
census_tracks_mn.to_crs(pp_mn_geo.crs, inplace = True)
acs_mn_data_geo = census_tracks_mn[['AFFGEOID','geometry']].merge(acs_mn_data,
                                                                  right_on = 'GEO.id',
                                                                  left_on = 'AFFGEOID', 
                                                                  how = 'right')
ax = acs_mn_data_geo.plot(color = 'cyan', alpha = 0.2, edgecolor = 'red', figsize=(10,10))
pp_mn_geo.plot(ax= ax, facecolor = 'None', edgecolor = 'black')
ax.set_title('Geospatial mapping of police precinct and census tracks',fontsize=16)
ax.set_axis_off()
# area for geodf with units in degrees is very small number. We need to get % split to be applied by taking the ratio.
# This can cause big rounding errors. So side effect of converting degrees to meters is sq.meters is much bigger number, 
# reducing the rounding error.

pp_mn_geo.to_crs({'init': 'epsg:32118', 'no_defs': True},inplace = True)
pp_mn_geo['area_pp'] = pp_mn_geo['geometry'].area
acs_mn_data_geo.to_crs({'init': 'epsg:32118', 'no_defs': True},inplace = True)
acs_mn_data_geo['area_acs'] = acs_mn_data_geo['geometry'].area
# overlay census polygons on police precincts. Overlap area / census track area will give split % to be applied to 
# acs data before rolling up.
##########################################################################
# pp_acs_mn = gpd.overlay(pp_mn_geo[['gridnum','geometry','area_pp']], 
#                           acs_mn_data_geo[['AFFGEOID','geometry','area_acs']], 
#                           how = 'intersection')
#
###################################################################
fpath = '../input/tempshapefile/pp_acs_mn.shp'
pp_acs_mn = gpd.read_file(fpath)
pp_acs_mn.head()
pp_acs_mn['common_area'] = pp_acs_mn['geometry'].area
pp_acs_mn['overlap_per'] = pp_acs_mn['common_area']/pp_acs_mn['area_acs']*100 
pp_acs_mn = pp_acs_mn.filter(items = ['gridnum','AFFGEOID','overlap_per'])
pp_acs_mn.head(3)
# acs data merged with pp and asc overlap dataframe.

pp_by_acs_data = pp_acs_mn.merge(acs_mn_data, left_on = 'AFFGEOID', right_on = 'GEO.id', how = 'left')
t = pp_by_acs_data.iloc[:,0:4]
for col in pp_by_acs_data.columns[4:]:
    t[col] = pp_by_acs_data[col] * pp_by_acs_data['overlap_per']

t.drop(columns = ['AFFGEOID','GEO.id','overlap_per'],inplace=True)
t = t.groupby(by='gridnum').sum().reset_index()
# calculate % for the populations rolled up by police precinct.

t['pop_male_per_rollup'] = t['male_pop']/t['tot_pop']*100
t['pop_female_per_rollup'] = t['female_pop']/t['tot_pop']*100
t['race_white_rollup'] = t['race_white']/t['tot_pop']*100
t['race_black_rollup'] = t['race_black']/t['tot_pop']*100
t['race_asian_rollup'] = t['race_asian']/t['tot_pop']*100
t['edu_less_educated_rollup'] = t['edu_pop_less']/t['edu_pop_tot']*100
t['pov_below_poverty_rollup'] = t['pov_pop_below']/t['pov_pop_tot']*100
t['emp_unemployment_rollup'] = t['pop_emp_unemployed_count']/t['pop_emp_tot']*100
t['ooh_owneroccupied_rollup'] = t['owner_occupied_units']/t['occupied_units']*100
t.drop(columns = pp_by_acs_data.columns[4:],inplace=True)
acs_police_record_by_pp = pp_pr_summary_mn.merge(t, on = 'gridnum')
bias_df = acs_police_record_by_pp[['gridnum','geometry','center']].copy()
# gender bias
bias_df = pd.merge(bias_df,acs_police_record_by_pp[['gridnum','male_stop_per','pop_male_per_rollup']], 
                  on = 'gridnum')
bias_df['male_bias'] = acs_police_record_by_pp['male_stop_per'] - acs_police_record_by_pp['pop_male_per_rollup']

bias_df = pd.merge(bias_df,acs_police_record_by_pp[['gridnum','female_stop_per','pop_female_per_rollup']], 
                  on = 'gridnum')
bias_df['female_bias'] = acs_police_record_by_pp['female_stop_per'] - acs_police_record_by_pp['pop_female_per_rollup']
# race bias
bias_df = pd.merge(bias_df,acs_police_record_by_pp[['gridnum','white_stop_per','race_white_rollup']], 
                  on = 'gridnum')
bias_df['white_bias'] = acs_police_record_by_pp['white_stop_per'] - acs_police_record_by_pp['race_white_rollup']
bias_df = pd.merge(bias_df,acs_police_record_by_pp[['gridnum','black_stop_per','race_black_rollup']], 
                  on = 'gridnum')
bias_df['black_bias'] = acs_police_record_by_pp['black_stop_per'] - acs_police_record_by_pp['race_black_rollup']
bias_df = pd.merge(bias_df,acs_police_record_by_pp[['gridnum','asian_stop_per','race_asian_rollup']], 
                  on = 'gridnum')
bias_df['asian_bias'] = acs_police_record_by_pp['asian_stop_per'] - acs_police_record_by_pp['race_asian_rollup']
bias_df['no_reason_stop'] = acs_police_record_by_pp['no_reason_stop']

for col in np.arange(5,len(bias_df.columns),3):
    choropleth_map(bias_df,bias_df.columns[col],fsize = (8,8))
    t = bias_df.iloc[:,[0,col-2,col-1,col]].sort_values(by = bias_df.columns[col],ascending = False)
    cleveland_plot(xmins = t.iloc[:,2], 
               xmaxs=t.iloc[:,1], 
               ylabs =  t.gridnum, 
               chart_title =  'Top bias police precincts', 
               xmin_lab =  t.columns[2] + ' %', 
               xmax_lab =  t.columns[1] + ' %',
               ytitle = 'Police Grid number',
               xtitle = 'percent',
               fsize=(6,6))   
    t = bias_df.iloc[:,[0,col-2,col-1,col]].sort_values(by = bias_df.columns[col],ascending = True)
    cleveland_plot(xmins = t.iloc[:,2], 
               xmaxs=t.iloc[:,1], 
               ylabs =  t.gridnum, 
               chart_title =  'Bottom bias police precincts', 
               xmin_lab =  t.columns[2] + ' %', 
               xmax_lab =  t.columns[1] + ' %',
               ytitle = 'Police Grid number',
               xtitle = 'percent',
               fsize=(6,6))       
acs_police_record_by_pp.columns
bias_df.head()

relation_df = acs_police_record_by_pp.filter(items = ['gridnum',
                                                      'geometry',
                                                      'center',
                                                      'total_stops',
                                                      'edu_less_educated_rollup',
                                                      'pov_below_poverty_rollup',
                                                     'emp_unemployment_rollup',
                                                     'ooh_owneroccupied_rollup'])
relation_df.head(3)
relation_df = relation_df.merge(bias_df[['gridnum','male_bias','female_bias','white_bias','black_bias','asian_bias','no_reason_stop']],on = 'gridnum',how = 'left')
relation_df.iloc[:,3:].corr()
ax = relation_df.plot(edgecolor = 'gray', 
                      figsize = (15,15), 
                      column = 'edu_less_educated_rollup', 
                      scheme = 'quantiles',
                     legend = True,
                     cmap = 'Blues')
plt.scatter(x = relation_df['center'].apply(lambda x: Point(x).x),
           y = relation_df['center'].apply(lambda x: Point(x).y),
           s = relation_df['total_stops']/12,
           edgecolor = 'red',
           color = 'None')
ax.set_title('Less education choropleth with bubble plot of total plots', fontsize=16)
ax.set_axis_off()
ax = relation_df.plot(edgecolor = 'gray', 
                      figsize = (15,15), 
                      column = 'pov_below_poverty_rollup', 
                      scheme = 'quantiles',
                     legend = True,
                     cmap = 'Greens')
plt.scatter(x = relation_df['center'].apply(lambda x: Point(x).x),
           y = relation_df['center'].apply(lambda x: Point(x).y),
           s = relation_df['total_stops']/12,
           edgecolor = 'red',
           color = 'None')
ax.set_title('Below poverty level choropleth with bubble plot of total plots', fontsize=16)
ax.set_axis_off()
ax = relation_df.plot(edgecolor = 'gray', 
                      figsize = (15,15), 
                      column = 'emp_unemployment_rollup', 
                      scheme = 'quantiles',
                     legend = True,
                     cmap = 'Greys')
plt.scatter(x = relation_df['center'].apply(lambda x: Point(x).x),
           y = relation_df['center'].apply(lambda x: Point(x).y),
           s = relation_df['total_stops']/12,
           edgecolor = 'red',
           color = 'None')
ax.set_title('Unemployment choropleth with bubble plot of total plots', fontsize=16)
ax.set_axis_off()
ax = relation_df.plot(edgecolor = 'gray', 
                      figsize = (15,15), 
                      column = 'ooh_owneroccupied_rollup', 
                      scheme = 'quantiles',
                     legend = True,
                     cmap = 'Blues')
plt.scatter(x = relation_df['center'].apply(lambda x: Point(x).x),
           y = relation_df['center'].apply(lambda x: Point(x).y),
           s = relation_df['total_stops']/12,
           edgecolor = 'red',
           color = 'None')
ax.set_title('Owner occupied housing choropleth with bubble plot of total plots', fontsize=16)
ax.set_axis_off()
ax = relation_df.plot(edgecolor = 'gray', 
                      figsize = (15,15), 
                      column = 'no_reason_stop', 
                      scheme = 'quantiles',
                     legend = True,
                     cmap = 'Blues')
plt.scatter(x = relation_df['center'].apply(lambda x: Point(x).x),
           y = relation_df['center'].apply(lambda x: Point(x).y),
           s = relation_df['total_stops']/12,
           edgecolor = 'red',
           color = 'None')
ax.set_title('No reason stops choropleth with bubble plot of total stops', fontsize=16)
ax.set_axis_off()
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
scale = MinMaxScaler()
relation_df_scaled = relation_df.iloc[:,3:8]
relation_df_scaled = scale.fit_transform(relation_df_scaled)
ranges = np.arange(2,10)
inert = []
for k in ranges:
    model = KMeans(n_clusters = k)
    model.fit(relation_df_scaled)
    inert.append(model.inertia_)

plt.plot(ranges,inert,'-o')
plt.xticks(ranges)
plt.title('Elbow plot', fontsize = 16, loc = 'center')
plt.show()
good_model = KMeans(n_clusters =6)
good_model.fit(relation_df_scaled)
relation_df['cluster_scaled'] = good_model.predict(relation_df_scaled)
pca = PCA(n_components = 2)
t = pca.fit_transform(relation_df_scaled)
pca_df = pd.DataFrame(data = t, columns = ['PCA1','PCA2'])
pca_df = pd.concat([pca_df,relation_df['cluster_scaled']],axis=1)
cluster_color = {0:'Clust-0',1:'Clust-1',2:'Clust-2',3:'Clust-3',4:'Clust-4',5:'Clust-5'}
pca_df['clust_color'] = pca_df.cluster_scaled.map(cluster_color)
clset = set(zip( pca_df.cluster_scaled,pca_df.clust_color ))
fig, ax = plt.subplots(figsize= (8,8))
sc = plt.scatter(pca_df.PCA1,pca_df.PCA2,c = pca_df.cluster_scaled,s=60, alpha = 1,cmap = 'tab20_r')
plt.title('Clustering of the police precincts', fontsize = 18)
plt.xlabel('PCA 1', fontsize =16)
plt.ylabel('PCA 2', fontsize =16)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
handles = [plt.plot([],color=sc.get_cmap()(sc.norm(c)),ls="", marker="o")[0] for c,l in clset ]
labels = [l for c,l in clset]
ax.legend(handles, labels)
ax.get_legend().set_bbox_to_anchor((1.3,1))
ax.get_legend().set_title('cluster-color')
fig, ax = plt.subplots(figsize= (8,8))
sc = plt.scatter(pca_df.PCA1,pca_df.PCA2,c = pca_df.cluster_scaled,s=60, alpha = 0.7,cmap = 'tab20_r')
plt.title('Clustering of the police precincts', fontsize = 18)
plt.xlabel('PCA 1', fontsize =16)
plt.ylabel('PCA 2', fontsize =16)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
handles = [plt.plot([],color=sc.get_cmap()(sc.norm(c)),ls="", marker="o")[0] for c,l in clset ]
labels = [l for c,l in clset]
ax.legend(handles, labels)
ax.get_legend().set_bbox_to_anchor((1.3,1))
ax.get_legend().set_title('cluster-color')

for i, pp in enumerate(relation_df.gridnum):
    ax.annotate(pp,(pca_df.PCA1[i]+0.01,pca_df.PCA2[i]-0.01),fontsize=10)
ax = relation_df.plot(column = 'cluster_scaled', figsize = (14,14),edgecolor = 'grey', legend = True, categorical = True, cmap = 'tab20_r')
ax.set_axis_off()
ax.set_title('Police Precincts clustered', fontsize = 18)
ax.get_legend().set_title('cluster-color')
ax.get_legend().set_bbox_to_anchor((1.1,1))
for i, pp in enumerate(relation_df.gridnum):
    ax.annotate(pp,(relation_df.geometry[i].centroid.x, relation_df.geometry[i].centroid.y))
plt.title('Correlation map for overall area', fontsize = 16)
sns.heatmap(relation_df.iloc[:,3:-1].corr(),vmin=-1, vmax=1, annot = True,cmap = 'Blues',fmt = '.1g')
for i in np.arange(6):
    plt.figure()
    plt.title('Corr plot Cluster ' + str(i), fontsize=14)
    sns.heatmap(relation_df[relation_df.cluster_scaled == i].iloc[:,3:-1].corr(),vmin=-1, vmax=1, annot = True,cmap = 'Blues',fmt = '.1g')