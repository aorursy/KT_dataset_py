import pandas as pd

import numpy as np

import seaborn as sns

from matplotlib import pyplot as plt

from scipy import stats

pd.set_option('display.max_columns',None)



df_site = pd.read_csv(r"../input/ground-water-levels-colorado/SITE_INFO.csv")

df_waterlevel= pd.read_csv(r"../input/ground-water-levels-colorado/WATERLEVEL.csv",dtype={"Comment": "string", "Original Direction": "string"})





df_site.info()

df_waterlevel.info()

# change type of select series to numeric

df_waterlevel['Water level in feet relative to NAVD88'] = pd.to_numeric(df_waterlevel['Water level in feet relative to NAVD88'],errors='coerce')

df_waterlevel['Depth to Water Below Land Surface in ft.'] = pd.to_numeric(df_waterlevel['Depth to Water Below Land Surface in ft.'],errors='coerce')

# Drop columns from waterlevel and site dfs

df_waterlevel= df_waterlevel.drop(columns=['AgencyCd', 'Original Parameter', 'Accuracy Value', 'Original Value', 'Data Provided by', 'Unnamed: 14','Original Direction', 'Original Unit','Accuracy Unit', 'Comment','Observation Method'])

df_site = df_site.drop(columns=['AgencyCd','SiteName','HorzDatum', 'Link', 'AgencyNm','WlSysName','AltAcy','AltUnitsNm','AltMethod','WellDepthUnitsNm','HorzMethod','HorzAcy','AltUnits','AltDatumCd','WellDepthUnits','NatAquiferCd','CountryCd','CountryNm','StateCd','StateNm','CountyCd','LocalAquiferCd','SiteType','QwSnFlag', 'QwSnDesc','QwBaselineFlag','QwBaselineDesc','WlWellChars','WlWellCharsDesc','WlWellType','WlWellPurposeDesc','WlWellPurposeNotes','LithDataProvider','ConstDataProvider','WlWellPurpose','WlWellTypeDesc','WlBaselineDesc','WlBaselineFlag','WlSnDesc','WlSnFlag'])



# Change errors in string objects

df_site['NatAqfrDesc']= df_site['NatAqfrDesc'].replace('Rio Grande aquifersystem', 'Rio Grande aquifer system') 

df_site['LocalAquiferName']= df_site['LocalAquiferName'].replace('Ogallala', 'Ogallala aquifer')

df_site['LocalAquiferName']= df_site['LocalAquiferName'].replace('Ogallala Formation', 'Ogallala aquifer')

df_site['well'] = 'well'



# rename select series

df_waterlevel = df_waterlevel.rename(columns={'Depth to Water Below Land Surface in ft.':'water_depth'})

df_waterlevel = df_waterlevel.rename(columns={'Water level in feet relative to NAVD88':'water_level_elev'})



# convert water depth to negative value

df_waterlevel['water_depth'] = df_waterlevel['water_depth']*-1



# df_site['county'] = df_site["CountyNm"].replace('County', '')

df_site['state'] = df_site["StateNm"]="CO"

df_waterlevel.info()

df_site.info()
df_site.head()
# Parse date to just year,month,day 

df_waterlevel['DateEdit'] = df_waterlevel['Time'].str[0:10]

print( df_waterlevel['Time'],df_waterlevel['DateEdit'])
# converting date to datetime series

df_waterlevel['DateEdit'] = pd.to_datetime(df_waterlevel['DateEdit'], format='%Y %m %d', errors = 'coerce')

# Merge datasheets on SiteNo unique identifier

df = pd.merge(df_waterlevel, df_site, on = 'SiteNo', how = 'outer')

df.head()

# df.describe()

# Locate first and last water measurment of water depth for each SiteNo.  Then merge with df.

base_depth_df=df.groupby(['SiteNo'])['water_depth'].nth(0)

latest_depth_df= df.groupby(['SiteNo'])['water_depth'].nth(-1)



df = pd.merge(df,base_depth_df, on = 'SiteNo', how = 'inner')

df= pd.merge(df,latest_depth_df, on = 'SiteNo', how = 'inner')



df.info()
# Number of wellsites prior to date filter

print('Total sites in dataset (all dates):',df.SiteNo.nunique())
# Locate earliest and latest date for each SiteNo



earliest_date_df= df.groupby(['SiteNo'])['DateEdit'].nth(0)

latest_date_df= df.groupby(['SiteNo'])['DateEdit'].nth(-1)



# Merge earliest and latest dates to df

df = pd.merge(df,earliest_date_df, on = 'SiteNo', how = 'inner')

df= pd.merge(df,latest_date_df, on = 'SiteNo', how = 'inner')



# Rename date start and date end

df = df.rename(columns={'DateEdit_y':'datestart'})

df = df.rename(columns={'DateEdit':'dateend'})

df = df.rename(columns={'DateEdit_x':'DateEdit'})



# Apply df filter to isolate wells with data starting prior to 1981 and ending later than 2019



df = df[(df['datestart']< "1991-01-01") & (df['dateend'] > "2019-01-01")]

print('Total sites with data prior to 1981 and after 2018:', df.SiteNo.nunique())

df.info()
# Add delta depth and total depth change columns



df['delta_base_level']= df['water_depth_x']-df['water_depth_y']

df['total_delta']=df['water_depth']-df['water_depth_y']



# Remove 1 outlier

remove_outliers = df['total_delta'] < 150

df = df[remove_outliers]

# Create 4 dataframes filtered by decade

df_80_90 = df[df['DateEdit']< "1990-01-01"]

df_90_00 = df[(df['DateEdit']>= "1990-01-01") & (df['DateEdit']< "2000-01-01")]

df_00_10 = df[(df['DateEdit']>= "2000-01-01") & (df['DateEdit']< "2010-01-01")]

df_10_20 = df[(df['DateEdit']>= "2010-01-01") & (df['DateEdit']< "2020-01-01")]



# Total number of measurments for all Sites for each decade

print('1980-1990 measurements:',len(df_80_90[df_80_90.delta_base_level > -500]))

print('1990-2000 measurements:', len(df_90_00[df_90_00.delta_base_level>=-500]))

print('2000-2010 measurements:', len(df_00_10[df_00_10.delta_base_level>=-500]))

print('2010-2020 measurements:', len(df_10_20[df_10_20.delta_base_level>=-500]))
# Historgram plot of change in water levels sorted by decade



site_80_90= df_80_90.groupby([pd.Grouper(key ='DateEdit',freq = '1Y'), 'SiteNo']).delta_base_level.mean().reset_index()

ax = sns.distplot(site_80_90['delta_base_level'], bins=50)

site_90_00= df_90_00.groupby([pd.Grouper(key ='DateEdit',freq = '1Y'), 'SiteNo']).delta_base_level.mean().reset_index()

ax = sns.distplot(site_90_00['delta_base_level'], bins=50)

site_00_10= df_00_10.groupby([pd.Grouper(key ='DateEdit',freq = '1Y'), 'SiteNo']).delta_base_level.mean().reset_index()

ax = sns.distplot(site_00_10['delta_base_level'], bins=50)

site_10_20= df_10_20.groupby([pd.Grouper(key ='DateEdit',freq = '1Y'), 'SiteNo']).delta_base_level.mean().reset_index()

ax = sns.distplot(site_00_10['delta_base_level'], bins=50)

# Stats for change in water level for each decade

print('1980-1990:',stats.describe(df_80_90['delta_base_level']))

# print('1990-2000:',stats.describe(df_90_00['delta_base_level']))

# print('2000-2010:',stats.describe(df_00_10['delta_base_level']))

print('2010-2020:',stats.describe(df_10_20['delta_base_level']))
# Calculate difference in means with 95% confidence



import math

def get_95_ci(array_1, array_2):

    sample_1_n = array_1.shape[0]

    sample_2_n = array_2.shape[0]

    sample_1_mean = array_1.mean()

    sample_2_mean = array_2.mean()

    sample_1_var = array_1.var()

    sample_2_var = array_2.var()

    mean_difference = sample_2_mean - sample_1_mean

    std_err_difference = math.sqrt((sample_1_var/sample_1_n)+(sample_2_var/sample_2_n))

    margin_of_error = 1.96 * std_err_difference

    ci_lower = mean_difference - margin_of_error

    ci_upper = mean_difference + margin_of_error

    return("The difference in means at the 95% confidence interval (two-tail) is between "+str(ci_lower)+" and "+str(ci_upper)+" feet.")

get_95_ci(df_80_90['delta_base_level'], df_10_20['delta_base_level'])

# Comparing water levels changes from 1980-1990 to water level changes 2010-2020

stats.ttest_ind(df_80_90['delta_base_level'], df_10_20['delta_base_level'])

# This T-Test is > 1.96 and the p-value is < .05, indicating there is significant difference in sample distributions
print(stats.shapiro(df_80_90['delta_base_level']))

print(stats.shapiro(df_10_20['delta_base_level']))
stats.kruskal(df_80_90['delta_base_level'], df_10_20['delta_base_level'])

# The point plot below shows, on average, ground water levels began to decrease in level between 2000-2010, and have continued to decrease to 2020

g = sns.pointplot(data=[df_80_90['delta_base_level'],df_90_00['delta_base_level'],df_00_10['delta_base_level'],

                        df_10_20['delta_base_level']],)

plt.title('Mean Change/Decade in Water Level from 1980 to 2020')

g.set(xticklabels = ['1980-1990','1990-2000','2000-2010', '2010-2020'])
# Group by site and take mean value for every 2 years

site_group= df.groupby([pd.Grouper(key = 'DateEdit', freq='2Y'), 'SiteNo']).delta_base_level.mean().reset_index()
fig = site_group[site_group.delta_base_level>=-500][

    ['DateEdit', 'delta_base_level']].groupby('DateEdit').mean().plot()

plt.title('Mean Change in Water Level from 1980 to 2020')

plt.show()
# View ground water change from 1980-2020 for all well sites

ax = site_group.pivot('DateEdit','SiteNo','delta_base_level').plot()

ax.legend_.remove()



plt.gcf().set_size_inches(14,8)

ax.axhline(250, color ='gray')

ax.axhline(200, color ='gray')

ax.axhline(150, color ='gray')

ax.axhline(100,color ='gray')

ax.axhline(50, color ='gray')

ax.axhline(0, color ='black')

ax.axhline(-50,color ='gray')

ax.axhline(-100, color ='gray')

ax.axhline(-150,color ='gray')

ax.axhline(-200,color ='gray')

plt.title('Change in Ground Water Levels/Well Site 1980 to 2020')

# Group by Geographic Aquifer System

aqfr_group = df.groupby([pd.Grouper(key = 'DateEdit', freq='2Y'), 'NatAqfrDesc']).delta_base_level.mean().reset_index()# Plot change in ground water levels 1980-2020 grouped by Geographic Aquifer System

ax =aqfr_group.pivot('DateEdit','NatAqfrDesc','delta_base_level').plot()

plt.gcf().set_size_inches(14,8)

ax.axhline(0, color ='gray')

ax.axhline(-5, color ='gray')

ax.axhline(-10, color ='gray')

ax.axhline(-15, color ='gray')

ax.axhline(5, color ='gray')

ax.axhline(10, color ='gray')

ax.axhline(15, color ='gray')

ax.axhline(-20, color ='gray')



aquifer= df.groupby(['NatAqfrDesc']).total_delta.describe().reset_index()

aquifer.head(30)


# Group by aquifer formation

aqfr_formation_group = df.groupby([pd.Grouper(key = 'DateEdit', freq='2Y'), 'LocalAquiferName']).delta_base_level.mean().reset_index()

aqfr_formation_group.groupby(['LocalAquiferName']).agg(['count'])





# # Plot change in ground water levels 1980-2020 grouped by aquifer formation

ax =aqfr_formation_group.pivot('DateEdit','LocalAquiferName','delta_base_level').plot()

plt.gcf().set_size_inches(12,6)

plt.ylabel('Delta Water Level (ft)')

plt.xlabel('Date')



ax.axhline(0, color ='black')

ax.axhline(-20, color ='gray')

ax.axhline(-40, color ='gray')

ax.axhline(-60, color ='gray')

ax.axhline(20, color ='gray')

ax.axhline(40, color ='gray')

ax.axhline(60, color ='gray')





# Total Change in Ground Water Levels

aquifer_formation= df.groupby(['LocalAquiferName']).total_delta.describe().reset_index()

aquifer_formation.sort_values(by='mean', ascending = True)

# aquifer_formation.head(15)
# Group by Colorado County and plot change in ground water depth through time

county_group = df.groupby([pd.Grouper(key = 'DateEdit', freq='2Y'), 'CountyNm']).delta_base_level.mean().reset_index()

county_group.tail(10)



ax = county_group.pivot('DateEdit','CountyNm','delta_base_level').plot()

plt.gcf().set_size_inches(14,6)

plt.ylabel('Delta Water Level (ft)')

plt.xlabel('Date')

ax.axhline(0, color ='gray')

ax.axhline(25, color ='gray')

ax.axhline(-25, color ='gray')





county= df.groupby(['CountyNm']).total_delta.describe().reset_index()

county.sort_values(by='mean', ascending = True)

# county.head(30)



# Group by aquifer type and plot change in ground water depth through time for Confined and Unconfined aquifers

county_group = df.groupby([pd.Grouper(key = 'DateEdit', freq='2Y'), 'AquiferType']).delta_base_level.mean().reset_index()

county_group.tail(10)



ax=county_group.pivot('DateEdit','AquiferType','delta_base_level').plot()

plt.gcf().set_size_inches(14,8)

plt.ylabel('Delta Water Level (ft)')

plt.xlabel('Date')

ax.axhline(0, color ='black')

ax.axhline(-2, color ='gray')

ax.axhline(-4, color ='gray')

ax.axhline(-6, color ='gray')

ax.axhline(2, color ='gray')

ax.axhline(4, color ='gray')

ax.axhline(6, color ='gray')

ax.axhline(8, color ='gray')





aqftype= df.groupby(['AquiferType']).total_delta.describe().reset_index()

aqftype.head(30)

# Record first a last measurment depths for each well site

df_first= df.groupby(['SiteNo']).water_depth_x.first().reset_index()

df_last = df.groupby(['SiteNo']).water_depth_x.last().reset_index()

df_start = df.groupby(['SiteNo']).datestart.first().reset_index()



# Merge to df 



df_merge = pd.merge(df_first,df_last, on='SiteNo', how = 'inner')

df_merge = pd.merge(df_merge,df_start, on ='SiteNo', how = 'inner')

df_merge = pd.merge(df_merge, df_site, on='SiteNo', how = 'inner')

df_merge
# rename auto named columns

df_merge = df_merge.rename(columns={'water_depth_x_x':'water_depth_start'})

df_merge = df_merge.rename(columns={'water_depth_x_y':'water_depth_latest'})

df_merge['Total_Delta'] = df_merge['water_depth_latest']-df_merge['water_depth_start']

remove_outlier = df_merge['Total_Delta'] < 150

df_merge = df_merge[remove_outlier]

df_merge
plt.figure(figsize = (5,8))

ax=sns.boxplot(y="Total_Delta", data=df_merge)

ax = sns.swarmplot(y="Total_Delta", data=df_merge, color=".2")

plt.xlabel('All Wells')

plt.ylabel('Delta Water Level (ft)')

plt.title('Total Delta Water Depth From First to Last Measurement')

ax.set_xticklabels(ax.get_xticklabels(), rotation=90);

ax.axhline(0, color ='gray')

ax.axhline(25, color ='gray')

ax.axhline(50,color ='gray')

ax.axhline(75, color ='gray')

ax.axhline(-25, color ='gray')

ax.axhline(-50,color ='gray')

ax.axhline(-75, color ='gray')

ax.axhline(-100,color ='gray')

ax.axhline(100,color ='gray')



df_merge.groupby(['well']).Total_Delta.describe().reset_index()
plt.figure(figsize = (10,8))

ax=sns.boxplot(y="Total_Delta",x ="AquiferType", data=df_merge)

ax = sns.swarmplot(x="AquiferType", y="Total_Delta", data=df_merge, color=".2")

plt.xlabel('Aquifer Type')

plt.ylabel('Delta Water Level (ft)')

plt.title('Total Delta Water Depth From First to Last Measurement for Aquifer Type')

ax.set_xticklabels(ax.get_xticklabels(), rotation=90);

ax.axhline(0, color ='gray')

ax.axhline(50,color ='gray')

ax.axhline(-50,color ='gray')

ax.axhline(-100,color ='gray')

ax.axhline(100,color ='gray')



aquifer_type= df_merge.groupby(['AquiferType']).Total_Delta.describe().reset_index()

aquifer_type
plt.figure(figsize = (10,8))



median_order = df_merge.groupby(by=['NatAqfrDesc'])['Total_Delta'].median().sort_values().index



ax=sns.boxplot(y="Total_Delta",x ="NatAqfrDesc", data=df_merge, order=median_order)

ax = sns.swarmplot(x="NatAqfrDesc", y="Total_Delta", data=df_merge, order=median_order, color=".2")

plt.xlabel('Aquifer System')

plt.ylabel('Delta Water Level')

plt.title('Total Delta Water Depth From First to Last Measurement For Aquifer System')

ax.set_xticklabels(ax.get_xticklabels(), rotation=90);

ax.axhline(0, color ='gray')

ax.axhline(50,color ='gray')

ax.axhline(-50,color ='gray')

ax.axhline(-100,color ='gray')

ax.axhline(100,color ='gray')



aqfr_sys= df_merge.groupby(['NatAqfrDesc']).Total_Delta.describe().reset_index()

aqfr_sys


plt.figure(figsize = (15,10))



median_order= df_merge.groupby(by =['LocalAquiferName'])['Total_Delta'].median().sort_values().index



ax =sns.boxplot(x ="LocalAquiferName", y="Total_Delta", data=df_merge, order=median_order)

ax = sns.swarmplot(x="LocalAquiferName", y="Total_Delta", data=df_merge, order=median_order, color=".2")

plt.xlabel('Local Aquifer')

plt.ylabel('Delta Water by Aquifer')

plt.title('Delta Water Level by Aquifer')

ax.set_xticklabels(ax.get_xticklabels(), rotation=90);

ax.axhline(0, color ='gray')

ax.axhline(50,color ='gray')

ax.axhline(-50,color ='gray')

ax.axhline(-100,color ='gray')

ax.axhline(100,color ='gray')



aqfr_formation= df_merge.groupby(['LocalAquiferName']).Total_Delta.describe().reset_index()

aqfr_formation.sort_values(by='mean', ascending= True)

import seaborn as sns

from matplotlib import pyplot as plt



plt.figure(figsize = (15,5))

median_order= df_merge.groupby(by =['CountyNm'])['Total_Delta'].median().sort_values().index

ax=sns.boxplot(y="Total_Delta",x ="CountyNm", data=df_merge, order = median_order)

ax = sns.swarmplot(x="CountyNm", y="Total_Delta", data=df_merge, order = median_order, color=".2")

plt.xlabel('County')

plt.ylabel('Delta Water Level')

plt.title('Delta Water Level by County')

ax.set_xticklabels(ax.get_xticklabels(), rotation=90);

ax.axhline(0, color ='gray')

ax.axhline(50,color ='gray')

ax.axhline(-50,color ='gray')

ax.axhline(-100,color ='gray')

ax.axhline(100,color ='gray')



county= df_merge.groupby(['CountyNm']).Total_Delta.describe().reset_index()

county.sort_values('mean', ascending=True)

# county.head(30)
