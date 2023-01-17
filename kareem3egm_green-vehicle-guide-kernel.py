# import library 

# data analysis and wrangling

import pandas as pd

import numpy as np

import random as rnd



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
# load datasets

df_08 = pd.read_csv('../input/green-vehicle-guide-datafile/all_alpha_08.csv') 

df_18 = pd.read_csv('../input/green-vehicle-guide-datafile/all_alpha_18.csv')
# view 2008 dataset

df_08.head()
# view 2018 dataset

df_18.head()
df_08.shape
df_18.shape
df_08.columns.values
df_18.columns.values
# drop columns from 2008 dataset

df_08.drop(['Stnd', 'Underhood ID', 'FE Calc Appr', 'Unadj Cmb MPG'], axis=1, inplace=True)



# confirm changes

df_08.head(1)
# drop columns from 2018 dataset

df_18.drop(['Stnd', 'Stnd Description', 'Underhood ID', 'Comb CO2'], axis=1, inplace=True)



# confirm changes

df_18.head(1)
# rename Sales Area to Cert Region

df_08.rename(columns={'Sales Area': 'Cert Region'}, inplace=True)



# confirm changes

df_08.head(1)
# replace spaces with underscores and lowercase labels for 2008 dataset

df_08.rename(columns=lambda x: x.strip().lower().replace(" ", "_"), inplace=True)



# confirm changes

df_08.head(1)
# replace spaces with underscores and lowercase labels for 2018 dataset

df_18.rename(columns=lambda x: x.strip().lower().replace(" ", "_"), inplace=True)



# confirm changes

df_18.head(1)
# confirm column labels for 2008 and 2018 datasets are identical

df_08.columns == df_18.columns
# make sure they're all identical like this

(df_08.columns == df_18.columns).all()


# confirm only certification region is California

df_08['cert_region'].unique()

# confirm only certification region is California

df_08['cert_region'].unique()

# confirm only certification region is California

df_08['cert_region'].unique()
# confirm only certification region is California

df_18['cert_region'].unique()
# drop certification region columns form both datasets

df_08.drop('cert_region', axis=1, inplace=True)

df_18.drop('cert_region', axis=1, inplace=True)
df_08.shape
df_18.shape
# view missing value count for each feature in 2008

df_08.isnull().sum()
# view missing value count for each feature in 2018

df_18.isnull().sum()
# drop rows with any null values in both datasets

df_08.dropna(inplace=True)

df_18.dropna(inplace=True)
# checks if any of columns in 2008 have null values - should print False

df_08.isnull().sum().any()
# checks if any of columns in 2018 have null values - should print False

df_18.isnull().sum().any()
# print number of duplicates in 2008 and 2018 datasets

print(df_08.duplicated().sum())

print(df_18.duplicated().sum())
# drop duplicates in both datasets

df_08.drop_duplicates(inplace=True)

df_18.drop_duplicates(inplace=True)
# print number of duplicates again to confirm dedupe - should both be 0

print(df_08.duplicated().sum())

print(df_18.duplicated().sum())
# check value counts for the 2008 cyl column

df_08['cyl'].value_counts()
# Extract int from strings in the 2008 cyl column

df_08['cyl'] = df_08['cyl'].str.extract('(\d+)').astype(int)
# Check value counts for 2008 cyl column again to confirm the change

df_08['cyl'].value_counts()
# convert 2018 cyl column to int

df_18['cyl'] = df_18['cyl'].astype(int)
df_08[df_08.air_pollution_score == '6/4']
# First, let's get all the hybrids in 2008

hb_08 = df_08[df_08['fuel'].str.contains('/')]

hb_08
# hybrids in 2018

hb_18 = df_18[df_18['fuel'].str.contains('/')]

hb_18
# create two copies of the 2008 hybrids dataframe

df1 = hb_08.copy()  # data on first fuel type of each hybrid vehicle

df2 = hb_08.copy()  # data on second fuel type of each hybrid vehicle



# Each one should look like this

df1
# columns to split by "/"

split_columns = ['fuel', 'air_pollution_score', 'city_mpg', 'hwy_mpg', 'cmb_mpg', 'greenhouse_gas_score']



# apply split function to each column of each dataframe copy

for c in split_columns:

    df1[c] = df1[c].apply(lambda x: x.split("/")[0])

    df2[c] = df2[c].apply(lambda x: x.split("/")[1])
# this dataframe holds info for the FIRST fuel type of the hybrid

# aka the values before the "/"s

df1
# this dataframe holds info for the SECOND fuel type of the hybrid

# aka the values before the "/"s

df2
# combine dataframes to add to the original dataframe

new_rows = df1.append(df2)



# now we have separate rows for each fuel type of each vehicle!

new_rows
# drop the original hybrid rows

df_08.drop(hb_08.index, inplace=True)



# add in our newly separated rows

df_08 = df_08.append(new_rows, ignore_index=True)
# check that all the original hybrid rows with "/"s are gone

df_08[df_08['fuel'].str.contains('/')]
df_08.shape
# create two copies of the 2018 hybrids dataframe, hb_18

df1 = hb_18.copy()

df2 = hb_18.copy()
# list of columns to split

split_columns = ['fuel', 'city_mpg', 'hwy_mpg', 'cmb_mpg']



# apply split function to each column of each dataframe copy

for c in split_columns:

    df1[c] = df1[c].apply(lambda x: x.split("/")[0])

    df2[c] = df2[c].apply(lambda x: x.split("/")[1])
# append the two dataframes

new_rows = df1.append(df2)



# drop each hybrid row from the original 2018 dataframe

# do this by using Pandas drop function with hb_18's index

df_18.drop(hb_18.index, inplace=True)



# append new_rows to df_18

df_18 = df_18.append(new_rows, ignore_index=True)
# check that they're gone

df_18[df_18['fuel'].str.contains('/')]
df_18.shape
# convert string to float for 2008 air pollution column

df_08.air_pollution_score = df_08.air_pollution_score.astype(float)
# convert int to float for 2018 air pollution column

df_18.air_pollution_score = df_18.air_pollution_score.astype(float)
# convert mpg columns to floats

mpg_columns = ['city_mpg', 'hwy_mpg', 'cmb_mpg']

for c in mpg_columns:

    df_18[c] = df_18[c].astype(float)

    df_08[c] = df_08[c].astype(float)
# convert from float to int

df_08['greenhouse_gas_score'] = df_08['greenhouse_gas_score'].astype(int)
df_08.dtypes
df_18.dtypes
df_08.dtypes == df_18.dtypes
df_08.fuel.value_counts()
df_18.fuel.value_counts()
# how many unique models used alternative sources of fuel in 2008

alt_08 = df_08.query('fuel in ["CNG", "ethanol"]').model.nunique()

alt_08
# how many unique models used alternative sources of fuel in 2018

alt_18 = df_18.query('fuel in ["Ethanol", "Electricity"]').model.nunique()

alt_18
plt.bar(["2008", "2018"], [alt_08, alt_18])

plt.title("Number of Unique Models Using Alternative Fuels")

plt.xlabel("Year")

plt.ylabel("Number of Unique Models");
# total unique models each year

total_08 = df_08.model.nunique()

total_18 = df_18.model.nunique()

total_08, total_18
prop_08 = alt_08/total_08

prop_18 = alt_18/total_18

prop_08, prop_18
plt.bar(["2008", "2018"], [prop_08, prop_18])

plt.title("Proportion of Unique Models Using Alternative Fuels")

plt.xlabel("Year")

plt.ylabel("Proportion of Unique Models");
veh_08 = df_08.groupby('veh_class').cmb_mpg.mean()

veh_08
veh_18 = df_18.groupby('veh_class').cmb_mpg.mean()

veh_18
# how much they've increased by for each vehicle class

inc = veh_18 - veh_08

inc
# only plot the classes that exist in both years

inc.dropna(inplace=True)

plt.subplots(figsize=(8, 5))

plt.bar(inc.index, inc)

plt.title('Improvements in Fuel Economy from 2008 to 2018 by Vehicle Class')

plt.xlabel('Vehicle Class')

plt.ylabel('Increase in Average Combined MPG');
# smartway labels for 2008

df_08.smartway.unique()
# get all smartway vehicles in 2008

smart_08 = df_08.query('smartway == "yes"')
# explore smartway vehicles in 2008

smart_08.describe()
# smartway labels for 2018

df_18.smartway.unique()
# get all smartway vehicles in 2018

smart_18 = df_18.query('smartway in ["Yes", "Elite"]')
smart_18.describe()
top_08 = df_08.query('cmb_mpg > cmb_mpg.mean()')

top_08.describe()
top_18 = df_18.query('cmb_mpg > cmb_mpg.mean()')

top_18.describe()
# rename 2008 columns

df_08.rename(columns=lambda x: x[:10] + "_2008", inplace=True)
# view to check names

df_08.head()
# merge datasets

df = df_08.merge(df_18, left_on='model_2008', right_on='model', how='inner')
# view to check merge

df.head()
model_mpg = df.groupby('model').mean()[['cmb_mpg_2008', 'cmb_mpg']]
model_mpg.head()
model_mpg['mpg_change'] = model_mpg['cmb_mpg'] - model_mpg['cmb_mpg_2008']
model_mpg.head()
max_change = model_mpg['mpg_change'].max()

max_change
model_mpg[model_mpg['mpg_change'] == max_change]
idx = model_mpg.mpg_change.idxmax()

idx
model_mpg.loc[idx]
df.head()
from pandas_profiling import ProfileReport 



profile = ProfileReport( df, title='Pandas profiling report ' , html={'style':{'full_width':True}})



profile.to_notebook_iframe()