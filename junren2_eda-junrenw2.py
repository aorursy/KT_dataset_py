import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from dask.distributed import Client
import gc
import dask.dataframe as dd
import dask.distributed
from dask.diagnostics import ProgressBar

client = Client(n_workers=1, threads_per_worker=4, processes=False, memory_limit='12GB')
client
# List all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# load the faf data
faf_1997_df = dd.read_csv("/kaggle/input/transportation/FAF4.5.1_Reprocessed_1997_State.csv")
faf_2002_df = dd.read_csv("/kaggle/input/transportation/FAF4.5.1_Reprocessed_2002_State.csv")
faf_2007_df = dd.read_csv("/kaggle/input/transportation/FAF4.5.1_Reprocessed_2007_State.csv",blocksize=1e6)
faf_2012_df = dd.read_csv("/kaggle/input/transportation/FAF4.5.1_State_2012.csv")
faf_2017_df = dd.read_csv("/kaggle/input/transportation/FAF4.5.1_State_2017.csv")
#delete the tmiles and curval column
faf_1997_df = faf_1997_df.drop(['tons_1997','tmiles_1997','curval_1997'], axis= 1)
faf_2002_df = faf_2002_df.drop(['tons_2002','tmiles_2002','curval_2002'], axis= 1)
faf_2007_df = faf_2007_df.drop(['tons_2007','tmiles_2007','curval_2007'], axis = 1)
faf_2012_df = faf_2012_df.drop(['tons_2012','tmiles_2012'], axis =1)
faf_2017_df = faf_2017_df.drop(['tons_2017','tmiles_2017','curval_2017'], axis = 1)
#rename the value column
faf_1997_df = faf_1997_df.rename(columns = {'value_1997':'value'})
faf_2002_df = faf_2002_df.rename(columns = {'value_2002':'value'})
faf_2007_df = faf_2007_df.rename(columns = {'value_2007':'value'})
faf_2012_df = faf_2012_df.rename(columns = {'value_2012':'value'})
faf_2017_df = faf_2017_df.rename(columns = {'value_2017':'value'})
#add a new column named 'year'
faf_1997_df['year'] = 1997
faf_2002_df['year'] = 2002
faf_2007_df['year'] = 2007
faf_2012_df['year'] = 2012
faf_2017_df['year'] = 2017
# concatenate the faf data from 1997-2017
df = dd.concat([faf_1997_df,faf_2002_df,faf_2007_df,faf_2012_df,faf_2017_df],axis = 0)

#only one year situation
# df = faf_2007_df
del faf_1997_df
del faf_2002_df
del faf_2007_df
del faf_2012_df
del faf_2017_df
#merge the state statistics to the combined faf datasets
#income: income represents the expenditure level for each states
income_df = dd.read_csv("/kaggle/input/transportation/income.csv",usecols=['GeoFips','LineCode',"1997","2002","2007","2012","2017"])
income_df = income_df[income_df["LineCode"]==3]
income_df = income_df.drop(["LineCode"],axis=1)
income_df = income_df.melt(id_vars='GeoFips',var_name="Year",value_name="value")
income_df.columns = ['FIPS',"Year","value"]
#only keep the income data for 1997,2002,2007,2012 and 2017
#income_df = income_df.loc[income_df['Year']==2007,:]

#clean the FIPS column, delete the zero so that it's consistent with FIPS in FAF dataset
income_df['FIPS'] = income_df['FIPS'].astype(str)
income_df['FIPS'] = income_df['FIPS'].map(lambda x:x[0:-3]).astype(int)
income_df['Year'] = income_df['Year'].astype(int)
income_df= income_df[["FIPS","Year","value"]]
#delete the duplicate (it's a  wierd that the income dataframe give the results twice)
income_df = income_df.drop_duplicates(subset=['FIPS','Year'])
#merge income to df
df = df.merge(income_df,left_on=['dms_origst','year'],right_on=['FIPS','Year'],how='left')
df = df.rename(columns = {'value_x':'value','value_y':'income_dms_origst'})
df = df.drop('Year',axis=1)
df = df.merge(income_df,left_on=['dms_destst','year'],right_on=['FIPS','Year'],how='left')
df = df.rename(columns = {'value_x':'value','value_y':'income_dms_destst'})
df = df.drop('Year',axis=1)
df = df.drop(['FIPS_y','FIPS_x'],axis=1)
del income_df
gc.collect()

#read animal_total
animal_total_df = dd.read_csv("../input/transportation/animals_total.csv",usecols=["State ANSI", "Year","Data Item", "Value"])
animal_total_df = animal_total_df[animal_total_df['Data Item']=="ANIMAL TOTALS, INCL PRODUCTS - SALES, MEASURED IN $"]
animal_total_df["Value"]= animal_total_df["Value"].astype(str).str.strip()
animal_total_df = animal_total_df.mask(animal_total_df== "(D)",np.nan)
animal_total_df = animal_total_df.mask(animal_total_df== "(Z)",np.nan)
animal_total_df["Value"] = animal_total_df["Value"].astype(float)
#animal_total_df = animal_total_df[animal_total_df['Year']==2007]
animal_total_df = animal_total_df.groupby(['State ANSI','Year']).sum().reset_index().compute()
df = df.merge(animal_total_df,left_on=['dms_origst',"year"],right_on=['State ANSI',"Year"],how='left')
df = df.drop('Year',axis=1)
df = df.rename(columns = {'Value':'animaltotal_dms_origst'})
df = df.merge(animal_total_df,left_on=['dms_destst','year'],right_on=['State ANSI',"Year"],how='left')
df = df.drop('Year',axis=1)
df = df.rename(columns = {'Value':'animaltotal_dms_destst'})
df = df.drop(['State ANSI_x','State ANSI_y'],axis = 1)
del animal_total_df
gc.collect()
# read aquaculture  because there is no 2007 data we use the nearest year 2005
# aquaculture_df = dd.read_csv("../input/transportation/aquaculture.csv",usecols=["State ANSI", "Year", "Value"])
# aquaculture_df = aquaculture_df[aquaculture_df['Year']==2005].compute().groupby(['State ANSI']).sum().reset_index()
# df = df.merge(aquaculture_df,left_on=['dms_origst'],right_on=['State ANSI'],how='left')
# df = df.rename(columns = {'Value':'acquaculture_dms_origst'})
# df = df.merge(aquaculture_df,left_on=['dms_destst'],right_on=['State ANSI'],how='left')
# df = df.rename(columns = {'Value':'aquaculture_dms_destst'})
# df = df.drop(['State ANSI_x','State ANSI_y','Year_x','Year_y'],axis = 1)
# del aquaculture_df
# gc.collect()
#read barley
barley_df = dd.read_csv("../input/transportation/barley.csv",usecols=["State ANSI", "Year", "Value"])
barley_df["Value"]= barley_df["Value"].astype(str).str.strip()
barley_df = barley_df.mask(barley_df== "(D)",np.nan)
barley_df = barley_df.mask(barley_df== "(Z)",np.nan)
barley_df["Value"] = barley_df["Value"].astype(float)

#barley_df = barley_df[barley_df["Year"]==2007]
df = df.merge(barley_df,left_on=['dms_origst','year'],right_on=['State ANSI',"Year"],how='left')
df = df.rename(columns = {'Value':'barley_dms_origst'})
df = df.drop('Year',axis=1)
df = df.merge(barley_df,left_on=['dms_destst','year'],right_on=['State ANSI',"Year"],how='left')
df = df.rename(columns = {'Value':'barley_dms_destst'})
df = df.drop('Year',axis=1)
df = df.drop(['State ANSI_x','State ANSI_y'],axis = 1)
del barley_df
gc.collect()
#read corn
corn_df = dd.read_csv("../input/transportation/corn.csv",usecols=["State ANSI", "Year", "Value"])
corn_df["Value"]= corn_df["Value"].astype(str).str.strip()
corn_df = corn_df.mask(corn_df== "(D)",np.nan)
corn_df = corn_df.mask(corn_df== "(Z)",np.nan)
corn_df["Value"] = corn_df["Value"].astype(float)
df = df.merge(corn_df,left_on=['dms_origst','year'],right_on=['State ANSI','Year'],how='left')
df = df.rename(columns = {'Value':'corn_dms_origst'})
df = df.drop('Year',axis=1)
df = df.merge(corn_df,left_on=['dms_destst','year'],right_on=['State ANSI','Year'],how='left')
df = df.rename(columns = {'Value':'corn_dms_destst'})
df = df.drop('Year',axis=1)
df = df.drop(['State ANSI_x','State ANSI_y'],axis = 1)
del corn_df
gc.collect()
#read croptotal
crop_total_df = dd.read_csv("../input/transportation/crop_total.csv",usecols=["State ANSI", "Year", "Value"])
crop_total_df["Value"]= crop_total_df["Value"].astype(str).str.strip()
crop_total_df = crop_total_df.mask(crop_total_df== "(D)",np.nan)
crop_total_df = crop_total_df.mask(crop_total_df== "(Z)",np.nan)
crop_total_df["Value"] = crop_total_df["Value"].astype(float)
df = df.merge(crop_total_df,left_on=['dms_origst','year'],right_on=['State ANSI','Year'],how='left')
df = df.rename(columns = {'Value':'crop_total_dms_origst'})
df = df.drop('Year',axis=1)
df = df.merge(crop_total_df,left_on=['dms_destst','year'],right_on=['State ANSI','Year'],how='left')
df = df.rename(columns = {'Value':'crop_total_dms_destst'})
df = df.drop('Year',axis=1)
df = df.drop(['State ANSI_x','State ANSI_y'],axis = 1)
del crop_total_df
gc.collect()
#read honey
honey_df = dd.read_csv("../input/transportation/honey.csv",usecols=["State ANSI", "Year", "Value"])
honey_df["Value"]= honey_df["Value"].astype(str).str.strip()
honey_df = honey_df.mask(honey_df== "(D)",np.nan)
honey_df = honey_df.mask(honey_df== "(Z)",np.nan)
honey_df["Value"] = honey_df["Value"].astype(float)
df = df.merge(honey_df,left_on=['dms_origst','year'],right_on=['State ANSI','Year'],how='left')
df = df.rename(columns = {'Value':'honey_dms_origst'})
df = df.drop('Year',axis=1)
df = df.merge(honey_df,left_on=['dms_destst','year'],right_on=['State ANSI','Year'],how='left')
df = df.rename(columns = {'Value':'honey_dms_destst'})
df = df.drop('Year',axis=1)
df = df.drop(['State ANSI_x','State ANSI_y'],axis = 1)
del honey_df
gc.collect()
#read milk
milk_df = dd.read_csv("../input/transportation/milk.csv",usecols=["State ANSI", "Year", "Value"])
milk_df["Value"]= milk_df["Value"].astype(str).str.strip()
milk_df = milk_df.mask(milk_df== "(D)",np.nan)
milk_df = milk_df.mask(milk_df== "(Z)",np.nan)
milk_df["Value"] = milk_df["Value"].astype(float)
df = df.merge(milk_df,left_on=['dms_origst','year'],right_on=['State ANSI','Year'],how='left')
df = df.rename(columns = {'Value':'milk_dms_origst'})
df = df.drop('Year',axis=1)
df = df.merge(milk_df,left_on=['dms_destst','year'],right_on=['State ANSI','Year'],how='left')
df = df.rename(columns = {'Value':'milk_dms_destst'})
df = df.drop('Year',axis=1)
df = df.drop(['State ANSI_x','State ANSI_y'],axis = 1)
del milk_df
gc.collect()
#read oats
oats_df = dd.read_csv("../input/transportation/oats.csv",usecols=["State ANSI", "Year", "Value"])
oats_df["Value"]= oats_df["Value"].astype(str).str.strip()
oats_df = oats_df.mask(oats_df== "(D)",np.nan)
oats_df = oats_df.mask(oats_df== "(Z)",np.nan)
oats_df["Value"] = oats_df["Value"].astype(float)
df = df.merge(oats_df,left_on=['dms_origst','year'],right_on=['State ANSI','Year'],how='left')
df = df.rename(columns = {'Value':'oats_dms_origst'})
df = df.drop('Year',axis=1)
df = df.merge(oats_df,left_on=['dms_destst','year'],right_on=['State ANSI','Year'],how='left')
df = df.rename(columns = {'Value':'oats_dms_destst'})
df = df.drop('Year',axis=1)
df = df.drop(['State ANSI_x','State ANSI_y'],axis = 1)
del oats_df
gc.collect()
#read rice
rice_df = dd.read_csv("../input/transportation/rice.csv",usecols=["State ANSI", "Year", "Value"])
rice_df["Value"]= rice_df["Value"].astype(str).str.strip()
rice_df = rice_df.mask(rice_df== "(D)",np.nan)
rice_df = rice_df.mask(rice_df== "(Z)",np.nan)
rice_df["Value"] = rice_df["Value"].astype(float)
df = df.merge(rice_df,left_on=['dms_origst','year'],right_on=['State ANSI','Year'],how='left')
df = df.rename(columns = {'Value':'rice_dms_origst'})
df = df.drop('Year',axis=1)
df = df.merge(rice_df,left_on=['dms_destst','year'],right_on=['State ANSI','Year'],how='left')
df = df.rename(columns = {'Value':'rice_dms_destst'})
df = df.drop('Year',axis=1)
df = df.drop(['State ANSI_x','State ANSI_y'],axis = 1)
del rice_df
gc.collect()
#read rye
rye_df = dd.read_csv("../input/transportation/rye.csv",usecols=["State ANSI", "Year", "Value"])
rye_df["Value"]= rye_df["Value"].astype(str).str.strip()
rye_df = rye_df.mask(rye_df== "(D)",np.nan)
rye_df = rye_df.mask(rye_df== "(Z)",np.nan)
rye_df["Value"] = rye_df["Value"].astype(float)
df = df.merge(rye_df,left_on=['dms_origst','year'],right_on=['State ANSI','Year'],how='left')
df = df.rename(columns = {'Value':'rye_dms_origst'})
df = df.drop('Year',axis=1)
df = df.merge(rye_df,left_on=['dms_destst','year'],right_on=['State ANSI','Year'],how='left')
df = df.rename(columns = {'Value':'rye_dms_destst'})
df = df.drop('Year',axis=1)
df = df.drop(['State ANSI_x','State ANSI_y'],axis = 1)
del rye_df
gc.collect()
#read sorghum
sorghum_df = dd.read_csv("../input/transportation/sorghum.csv",usecols=["State ANSI", "Year", "Value"])
sorghum_df["Value"]= sorghum_df["Value"].astype(str).str.strip()
sorghum_df = sorghum_df.mask(sorghum_df== "(D)",np.nan)
sorghum_df = sorghum_df.mask(sorghum_df== "(Z)",np.nan)
sorghum_df["Value"] = sorghum_df["Value"].astype(float)
df = df.merge(sorghum_df,left_on=['dms_origst','year'],right_on=['State ANSI','Year'],how='left')
df = df.rename(columns = {'Value':'sorghum_dms_origst'})
df = df.drop('Year',axis=1)
df = df.merge(sorghum_df,left_on=['dms_destst','year'],right_on=['State ANSI','Year'],how='left')
df = df.rename(columns = {'Value':'sorghum_dms_destst'})
df = df.drop('Year',axis=1)
df = df.drop(['State ANSI_x','State ANSI_y'],axis = 1)
del sorghum_df
gc.collect()
#read wheat
wheat_df = dd.read_csv("../input/transportation/wheat.csv",usecols=["State ANSI", "Year", "Value"])
wheat_df["Value"]= wheat_df["Value"].astype(str).str.strip()
wheat_df = wheat_df.mask(wheat_df== "(D)",np.nan)
wheat_df = wheat_df.mask(wheat_df== "(Z)",np.nan)
wheat_df["Value"] = wheat_df["Value"].astype(float)
df = df.merge(wheat_df,left_on=['dms_origst','year'],right_on=['State ANSI','Year'],how='left')
df = df.rename(columns = {'Value':'wheat_dms_origst'})
df = df.drop('Year',axis=1)
df = df.merge(wheat_df,left_on=['dms_destst','year'],right_on=['State ANSI','Year'],how='left')
df = df.rename(columns = {'Value':'wheat_dms_destst'})
df = df.drop('Year',axis=1)
df = df.drop(['State ANSI_x','State ANSI_y'],axis = 1)
del wheat_df
gc.collect()
#read gdp
gdp_df = dd.read_csv("../input/transportation/state_gdp_1997_2018.csv",usecols=["GeoFIPS", "1997","2002","2007","2012","2017"])
gdp_df = gdp_df.melt(id_vars="GeoFIPS",var_name="Year",value_name="Value")
gdp_df['Year'] = gdp_df['Year'].astype(int)
df = df.merge(gdp_df,left_on=['dms_origst','year'],right_on=['GeoFIPS',"Year"],how='left')
df = df.rename(columns = {"Value":'gdp_dms_origst'})
df = df.merge(gdp_df,left_on=['dms_destst','year'],right_on=['GeoFIPS',"Year"],how='left')
df = df.rename(columns = {"Value":'gdp_dms_destst'})
df = df.drop(['GeoFIPS_x','GeoFIPS_y',"Year_x","Year_y"],axis = 1)
del gdp_df
gc.collect()
#compute the distance between the origination and destination
from haversine import haversine, Unit
 
state_location_df = pd.read_csv("/kaggle/input/transportation/state_distance.csv")
#create a distance dataframe
distance_df = pd.DataFrame(columns=['state1','state2','distance'])
for i in range(len(state_location_df)):
    state1 = state_location_df.loc[i,'FIPS']
    state1_location = (state_location_df.loc[i,'Latitude'],state_location_df.loc[i,'Longitude'])
    for j in range(len(state_location_df)):
        state2 = state_location_df.loc[j,'FIPS']
        state2_location = (state_location_df.loc[j,'Latitude'],state_location_df.loc[j,'Longitude'])
        distance_df = distance_df.append({'state1':state1,'state2':state2,'distance':haversine(state1_location,state2_location)},ignore_index=True)
df = df.merge(distance_df,left_on=['dms_origst','dms_destst'],right_on=['state1','state2'],how='left')   
df = df.drop(['state1','state2'],axis=1)
del distance_df
gc.collect()
#Here we set the fr_orig and fr_dest 0 for domestic trade

df['fr_orig'] = df['fr_orig'].mask(df['fr_orig'].isna(), 800)
df['fr_dest'] = df['fr_dest'].mask(df['fr_dest'].isna(), 800)
df.columns
#calculate the missing
missing_values = df.isna().sum()
with ProgressBar():
    percent_missing = ((missing_values/df.index.size)*100).compute()
percent_missing
columns_to_drop= ["rice_dms_origst","rice_dms_destst"]
df = df.drop(columns_to_drop,axis =1)
df = df.compute()
df.head()
df.columns
df.info()
corr = df.corr()
corr["value"]
f,ax = plt.subplots(figsize=(18,15))
#plot a heatmap
sns.heatmap(corr,cmap='Blues')
plt.title("Pearson correlation Matrix",fontsize=20)
plt.show()
#distribution for each varible
df.replace(np.nan, -9999)
hist = df.hist(figsize=(20,18))
df.columns
df_domestic = df[df["fr_orig"]==800]
df_foreign = df[df["fr_orig"]!=800]
corr = df_domestic.corr()
f,ax = plt.subplots(figsize=(18,15))
#plot a heatmap
sns.heatmap(corr,cmap='Blues')
plt.title("Pearson correlation Matrix",fontsize=20)
plt.show()
corr = df_foreign.corr()
f,ax = plt.subplots(figsize=(18,15))
#plot a heatmap
sns.heatmap(corr,cmap='Blues')
plt.title("Pearson correlation Matrix",fontsize=20)
plt.show()
#distribution for each varible

hist = df_domestic.hist(figsize=(20,18))

hist = df_foreign.hist(figsize=(20,18))
df_sctg1 = df[df["sctg2"]==1]
df_sctg2 = df[df["sctg2"]==2]
df_sctg3 = df[df["sctg2"]==3]
df_sctg4 = df[df["sctg2"]==4]
df_sctg5 = df[df["sctg2"]==5]
df_sctg6 = df[df["sctg2"]==6]
df_sctg7 = df[df["sctg2"]==7]


corr = df_sctg1.corr()
f,ax = plt.subplots(figsize=(18,15))
#plot a heatmap
sns.heatmap(corr,cmap='Blues')
plt.title("Pearson correlation Matrix",fontsize=20)
plt.show()
corr = df_sctg2.corr()
f,ax = plt.subplots(figsize=(18,15))
#plot a heatmap
sns.heatmap(corr,cmap='Blues')
plt.title("Pearson correlation Matrix",fontsize=20)
plt.show()
corr = df_sctg3.corr()
f,ax = plt.subplots(figsize=(18,15))
#plot a heatmap
sns.heatmap(corr,cmap='Blues')
plt.title("Pearson correlation Matrix",fontsize=20)
plt.show()
corr = df_sctg4.corr()
f,ax = plt.subplots(figsize=(18,15))
#plot a heatmap
sns.heatmap(corr,cmap='Blues')
plt.title("Pearson correlation Matrix",fontsize=20)
plt.show()
corr = df_sctg5.corr()
f,ax = plt.subplots(figsize=(18,15))
#plot a heatmap
sns.heatmap(corr,cmap='Blues')
plt.title("Pearson correlation Matrix",fontsize=20)
plt.show()
corr = df_sctg6.corr()
f,ax = plt.subplots(figsize=(18,15))
#plot a heatmap
sns.heatmap(corr,cmap='Blues')
plt.title("Pearson correlation Matrix",fontsize=20)
plt.show()
corr = df_sctg7.corr()
f,ax = plt.subplots(figsize=(18,15))
#plot a heatmap
sns.heatmap(corr,cmap='Blues')
plt.title("Pearson correlation Matrix",fontsize=20)
plt.show()
hist = df_sctg1.hist(figsize=(20,18))
hist = df_sctg2.hist(figsize=(20,18))
hist = df_sctg3.hist(figsize=(20,18))
hist = df_sctg4.hist(figsize=(20,18))
hist = df_sctg5.hist(figsize=(20,18))
hist = df_sctg6.hist(figsize=(20,18))
hist = df_sctg7.hist(figsize=(20,18))
