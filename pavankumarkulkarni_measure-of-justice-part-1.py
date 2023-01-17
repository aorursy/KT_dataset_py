#Load libraries needed for the solution
%matplotlib inline
import geopandas as gpd
from shapely.geometry import Point
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# read the census track data and inspect first few rows
#base_path = '../input/cpe-data/' for kaggle
base_path = 'input/'
census_tracks_mn = gpd.read_file(base_path+'mncensustracks/cb_2017_27_tract_500k.shp')
census_tracks_mn.head(2)
# Check the coordinate reference system. It is essential all geospatial data is in same CRS for doing any geospatial 
# data analysis.
census_tracks_mn.crs
police_precinct_mn = gpd.read_file(base_path +
                                   'Dept_24-00098/24-00098_Shapefiles/StPaul_geo_export_6646246d-0f26-48c5-a924-f5a99bb51c47.shp')
police_precinct_mn.head(2)
police_precinct_mn.crs
police_precinct_mn.to_crs({'init': 'epsg:32118', 'no_defs': True},inplace = True)
census_tracks_mn.to_crs({'init': 'epsg:32118', 'no_defs': True},inplace = True)
ax= census_tracks_mn.plot(facecolor = 'None', edgecolor = 'gray',figsize = (8,8))
ax.set_title('Fig 1 - MN state census and police precincts data provided', fontsize=18)
police_precinct_mn.plot(ax=ax)
ax.set_axis_off()
# get the GEO.ids from ACS datafile for joining with census tracks.
acs_mn = pd.read_csv(base_path + 'Dept_24-00098/24-00098_ACS_data/24-00098_ACS_education-attainment/ACS_15_5YR_S1501_with_ann.csv',
                    skiprows = [1],
                    usecols=[0])
acs_mn.head(2)
census_acs_mn = pd.merge(census_tracks_mn, acs_mn, how = 'inner', left_on = 'AFFGEOID', right_on = 'GEO.id')
ax= census_acs_mn.plot(facecolor = 'None', edgecolor = 'gray',figsize = (12,12))
police_precinct_mn.plot(ax=ax, alpha = 0.3,edgecolor='black',facecolor='red')
ax.set_title('Fig 2 - ACS data overlayed with police precinct data', fontsize=18)
ax.set_axis_off()
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
police_records_mn.to_crs(census_acs_mn.crs,inplace=True)
ax= census_acs_mn.plot(facecolor = 'None', edgecolor = 'black',figsize = (12,12))
police_precinct_mn.plot(ax=ax, alpha = 0.3,edgecolor='black')
police_records_mn.plot(ax = ax, color = 'red', alpha=0.3)
ax.set_title('Fig 3 -Vehicle stops overlayed on police precincts and ACS census', fontsize=18)
ax.set_axis_off()
police_precinct_mn['center'] = police_precinct_mn['geometry'].centroid 
pp_records_mn = gpd.sjoin(police_records_mn, police_precinct_mn[['gridnum','geometry']], how = 'inner', op = 'intersects')
pp_rec_count_mn = pd.DataFrame(pp_records_mn['gridnum'].value_counts())
pp_rec_count_mn.reset_index(inplace = True)
pp_rec_count_mn.columns = ['gridnum','crime_count']
police_precinct_mn = pd.merge(police_precinct_mn, pp_rec_count_mn, on = 'gridnum', how = 'left')
ax = police_precinct_mn.plot(facecolor = 'None', edgecolor = 'black',figsize = (12,12))
plt.scatter(x=police_precinct_mn['center'].apply(lambda x: Point(x).x),
           y=police_precinct_mn['center'].apply(lambda x: Point(x).y),
           s = police_precinct_mn['crime_count']/8,
           alpha=0.4,
           c='red')
ax.set_title('Fig - 4 Vehicle stops count in each police precinct', fontsize =18)
ax.set_axis_off()
# append area of each census track and police precincts to the df.
census_acs_mn['area_census'] = census_acs_mn['geometry'].area
police_precinct_mn['area_pp'] = police_precinct_mn['geometry'].area
# overlay census polygons on police precincts.
pp_census_mn = gpd.overlay(police_precinct_mn[['gridnum','geometry','area_pp']], 
                           census_acs_mn[['AFFGEOID','geometry','area_census']], 
                           how = 'intersection')
pp_census_mn['area_overlap'] = pp_census_mn['geometry'].area
pp_census_mn['percent_overlap'] = pp_census_mn['area_overlap']/pp_census_mn['area_census']*100
pp_census_mn.head(3)
pp_census_mn[pp_census_mn['gridnum'] == '280']
ax = pp_census_mn[pp_census_mn['gridnum'] == '280'].plot(figsize=(8,8),facecolor = 'None', edgecolor = 'black')
census_acs_mn[census_acs_mn['AFFGEOID'] == '1400000US27123042503'].plot(ax=ax, edgecolor='red',alpha=0.2)
census_acs_mn[census_acs_mn['AFFGEOID'] == '1400000US27123037403'].plot(ax=ax, edgecolor='red',alpha=0.2)
ax.set_title('Fig 5 - 280 precinct and overlapping census tracks')
ax.set_axis_off()
test = pd.DataFrame(pp_census_mn.groupby(by = ['gridnum'])['area_overlap'].sum())
test.reset_index(drop=False,inplace=True)
test1 = pd.merge(test,police_precinct_mn,on='gridnum')
test1['diff'] = (test1.area_pp-test1.area_overlap)/test1.area_pp*100
test1['diff'].max(),test1['diff'].min()
test1[test1['diff'] == test1['diff'].max()]
pp_census_mn[pp_census_mn['gridnum'] == '242']
ax = police_precinct_mn[police_precinct_mn['gridnum'] == '242'].plot(facecolor = 'None',edgecolor = 'black',figsize=(6,6))
census_acs_mn[census_acs_mn['AFFGEOID'] == '1400000US27123037601'].plot(ax=ax, edgecolor='red',alpha=0.2)
ax.set_title('Fig 6 - PP 242 and associated census tracks', fontsize=18)
ax.set_axis_off()
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
rsa_meta_mn
# Read race_sex_age data
rsa_mn_data = pd.read_csv(base_path + 'Dept_24-00098/24-00098_ACS_data/24-00098_ACS_race-sex-age/ACS_15_5YR_DP05_with_ann.csv',
           skiprows = [1], na_values = '-')

#census_acs_mn has census tracks which overlap on police precincts. We need to filter out any census data
#not in police precincts 
rsa_mn_data_train = census_acs_mn[['AFFGEOID']].merge(rsa_mn_data,left_on = 'AFFGEOID',right_on = 'GEO.id',how = 'inner')
# Filter ACS data for Gender bias i.e total population, male and female
all_est_columns = rsa_mn_data_train.columns[rsa_mn_data_train.columns.str.contains('HC01')]
rsa_mn_gender_data_train = rsa_mn_data_train.filter(items = ['GEO.id','HC01_VC03','HC01_VC04','HC01_VC05'])
rsa_mn_gender_data_train.head(2)
pp_census_mn_gender_train = pd.merge(pp_census_mn, rsa_mn_gender_data_train, 
                                     right_on = 'GEO.id', 
                                     left_on = 'AFFGEOID', 
                                     how = 'left')
pp_census_mn_gender_train['total_pop'] = pp_census_mn_gender_train.percent_overlap*pp_census_mn_gender_train.HC01_VC03
pp_census_mn_gender_train['male_pop'] = pp_census_mn_gender_train.percent_overlap*pp_census_mn_gender_train.HC01_VC04
pp_census_mn_gender_train['female_pop'] = pp_census_mn_gender_train.percent_overlap*pp_census_mn_gender_train.HC01_VC05
pp_mn_gender_train = pp_census_mn_gender_train[['gridnum','total_pop','male_pop','female_pop']].groupby(by = ['gridnum']).sum()
pp_mn_gender_train['male_per'] = pp_mn_gender_train['male_pop']/pp_mn_gender_train['total_pop']*100
pp_mn_gender_train['female_per'] = pp_mn_gender_train['female_pop']/pp_mn_gender_train['total_pop']*100
pp_mn_gender_geo_train = police_precinct_mn.merge(pp_mn_gender_train, on = 'gridnum', how='left')
ax = pp_mn_gender_geo_train.plot(column = 'female_per', legend = True)
ax.set_title('Female percentage population by police precinct')
ax.set_axis_off()
t = police_records_mn[['INCIDENT_DATE_YEAR','LOCATION_DISTRICT','SUBJECT_GENDER']].groupby(['LOCATION_DISTRICT','SUBJECT_GENDER']).count()
t.reset_index(inplace = True)
t.head(5)
t1 = t.pivot(index = 'LOCATION_DISTRICT',columns = 'SUBJECT_GENDER', values = 'INCIDENT_DATE_YEAR').reset_index()
t1['female_per_rec'] = t1.Female/(t1.Female + t1.Male)*100
t1['male_per_rec'] = t1.Male/(t1.Female + t1.Male)*100
t1['LOCATION_DISTRICT'] = t1.LOCATION_DISTRICT.astype(int)
t1.head(4)
# Make the 'gridnum' type same as location_district to allow merge.
pp_mn_gender_geo_train['gridnum'] = pp_mn_gender_geo_train['gridnum'].astype(int)
t3 = pp_mn_gender_geo_train.merge(t1, left_on = 'gridnum', right_on = 'LOCATION_DISTRICT')
(t3['male_per'] - t3['male_per_rec'] ).mean()
t3.male_per_rec.mean(),t3.female_per_rec.mean()
t3['male_bias'] = t3['male_per_rec'] - t3['male_per'] 
ax = t3.plot(column = 'male_bias',legend = True, figsize = (12,12),k=4, cmap = 'PuRd')
ax.set_title('Male gender bias for vehicle stoppages.\n% driver stopped:Male - % population:Male',fontsize=18)
plt.axis('equal')
ax.set_axis_off()

t3[t3.male_bias == t3.male_bias.max()]['gridnum']
t4 = t3[['gridnum','male_per','male_per_rec','male_bias']].copy()
t4.sort_values(by = 'male_bias', ascending = False, inplace = True)
t4.reset_index(drop=True, inplace = True)
t4.head()
tot_recs = 40
y_range = np.arange(1,tot_recs+1)

fig, ax = plt.subplots(figsize=(8,8))
plt.hlines(y = y_range, xmin = t4.male_per[:tot_recs], xmax = t4.male_per_rec[:tot_recs], 
           alpha=0.3, color='gray', linewidth = 2)
plt.scatter(x = t4.male_per[:tot_recs],  y = y_range, color = 'green', label = 'pop male%', s =50, marker='o')
plt.scatter(x = t4.male_per_rec[:tot_recs],  y = y_range, color = 'red', label = 'driver stop male%', s = 50,marker = 'H')
plt.yticks(y_range, t4.gridnum[:tot_recs],fontsize=12)
plt.ylabel('Police Grid Number', fontsize = 14, color = 'gray')
plt.ylim(tot_recs+1,0)
plt.xticks(fontsize=12)
plt.xlabel('Percentage', fontsize = 14, color = 'gray')
plt.legend(fontsize = 12,loc='center right')
plt.title('Male : Gender bias - Top '+str(tot_recs)+' police precincts', fontsize = 18)
plt.grid(False)
ax.set_facecolor('white')
fig.set_facecolor('w')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
