# importing Required Libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
# Alter display to Maximum 

pd.set_option('display.max_columns',None)

pd.set_option('display.max_rows',None)
# reading the data

dataframe=pd.read_csv('../input/power-plant-database/global_power_plant_database.csv')

dataframe.sample(5)
dataframe.shape
dataframe.isna().sum()
dataframe.describe().T
dataframe.info()
# Converting to Int64 to get rid of values after point

dataframe['commissioning_year']=dataframe['commissioning_year'].fillna(0).astype('int64')

# creating new Dataframe with only required values

df=dataframe.drop(['owner','source','url','geolocation_source','year_of_capacity_data'],axis=1)
df.sample(10)
for col in df.columns:

    print('Number of unique value in the ',col,'is ',df[col].nunique())
com_year=pd.DataFrame(df['commissioning_year'].value_counts(ascending=True))

com_year=com_year.drop(0,axis=0)

com_year.head(5)
sns.set_style("whitegrid")

plt.figure(figsize=(20,8))

plt.title('Number of Powerplants commissioned as per Year wise')

sns.scatterplot(com_year.index,com_year.commissioning_year,alpha=0.8)
pf_list=pd.DataFrame(df['primary_fuel'].value_counts())

of1_list=pd.DataFrame(df['other_fuel1'].value_counts())

of2_list=pd.DataFrame(df['other_fuel2'].value_counts())

of3_list=pd.DataFrame(df['other_fuel3'].value_counts())



pp_fuel=pd.concat([pf_list,of1_list,of2_list,of3_list],axis=1)

pp_fuel=pp_fuel.fillna(0)

pp_fuel['total']=pp_fuel['primary_fuel']+pp_fuel['other_fuel1']+pp_fuel['other_fuel2']+pp_fuel['other_fuel3']

pp_fuel
fig,axes=plt.subplots(2,3,figsize=(40,20))



axes[0,0].set_title('Primary Fuel')

axes[0,0].bar(pf_list.index,pf_list.primary_fuel)

axes[0,0].tick_params(axis='x', labelrotation=45)



axes[0,1].set_title('other fuel 1')

axes[0,1].bar(of1_list.index,of1_list.other_fuel1)

axes[0,1].tick_params(axis='x', labelrotation=45)



axes[0,2].set_title('other fuel 2')

axes[0,2].bar(of2_list.index,of2_list.other_fuel2)

axes[0,2].tick_params(axis='x', labelrotation=45)



axes[1,0].set_title('other fuel 3')

axes[1,0].bar(of3_list.index,of3_list.other_fuel3)

axes[1,0].tick_params(axis='x', labelrotation=45)



axes[1,1].set_title('Total')

axes[1,1].bar(pp_fuel.index,pp_fuel.total)

axes[1,1].tick_params(axis='x', labelrotation=45)

plt.show()

df.generation_gwh_2014=df.generation_gwh_2014.fillna(0)

min(df.generation_gwh_2014)
df[(df.generation_gwh_2014<0)&(df.generation_gwh_2015<0)&(df.generation_gwh_2016<0)&(df.generation_gwh_2017<0)].head(5)
df1=df[df['commissioning_year']>2009]

print('After 2009, {} Powerplants are comissioned across the world.'.format(df.shape[0]))
pp_2009=pd.DataFrame(df1.primary_fuel.value_counts())

print(pp_2009)

pp_2009.plot(kind='bar')
pp_2010=pd.DataFrame(df1.groupby('primary_fuel')['capacity_mw'].sum().sort_values(ascending=False))



pp_2010_sum=pp_2010['capacity_mw'].sum()

pp_2010['percentage']=pp_2010['capacity_mw']/pp_2010_sum*100

pp_2010
df2=df[(df['commissioning_year']>2002)&(df['commissioning_year']<2010)]

print('Number of Powerplants commissioned from 2002 to 2010 are',df2.shape[0],'.')



pp_2002=pd.DataFrame(df2.groupby('primary_fuel')['capacity_mw'].sum().sort_values(ascending=False))

pp_2002_sum=pp_2002['capacity_mw'].sum()

pp_2002['percentage']=pp_2002['capacity_mw']/pp_2002_sum*100



pp_2002
merge=pp_2010.join(pp_2002,lsuffix='_2018', rsuffix='_2009')

merge['%_inc']=(merge['capacity_mw_2018']-merge['capacity_mw_2009'])/merge['capacity_mw_2009']

merge
df1[df1['primary_fuel']=='Solar'].groupby(by='country_long')['capacity_mw'].sum().sort_values(ascending=False).head(5)
import geopandas as gpd

shapefile = '../input/world-shapefile/world_shapefile.shp'

#Read shapefile using Geopandas

gdf = gpd.read_file(shapefile)



plt.figure(figsize=(43,16))

gdf.plot(figsize=(43,16))

sns.scatterplot(df1['longitude'],df1['latitude'],hue=df1['primary_fuel'])

plt.title('PowerPlants Comissioned after 2009')

plt.show()
brazil_2009=df1[df1['country_long']=='Brazil']

brazil_2009.head(5)
brazil_shape=gdf[gdf['NAME']=='Brazil']['geometry']

brazil=gdf[gdf['NAME']=='Brazil']

brazil.plot(figsize=(20,20))

sns.scatterplot(brazil_2009['longitude'],brazil_2009['latitude'],hue=brazil_2009['primary_fuel'])

plt.title('PowerPlants Comissioned after 2009 in brazil')

plt.show()